import os
import csv
import pickle
import hashlib
import argparse
import itertools
import numpy as np
import faiss


# ============================================================
# Basic helpers
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def path_hash(path):
    return hashlib.md5(path.encode("utf-8")).hexdigest()[:10]


def cache_prefix(cache_dir, file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    h = path_hash(os.path.abspath(file_path))
    return os.path.join(cache_dir, f"{base}_{h}")


# ============================================================
# Original threshold logic preserved
# ============================================================
def select_threshold_func(tensor, percent):
    """
    Same logic as your original code:
    sort distances and pick the element at int(n * percent / 100).
    """
    tensor = np.sort(tensor)
    percent = percent / 100.0
    n = len(tensor)
    idx = int(n * percent)

    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0

    return float(tensor[idx])


def jaccard_set_similarity(set_a, set_b):
    union = set_a | set_b
    if len(union) == 0:
        return 1.0
    return len(set_a & set_b) / len(union)


# ============================================================
# Large text embedding loader with caching
# ============================================================
def count_rows_and_validate_cols(txt_path, cols, dtype=np.float32):
    n_rows = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            values = np.fromstring(line, sep=" ", dtype=dtype)
            if values.size != cols:
                raise ValueError(
                    f"[ERROR] Column mismatch in {txt_path} at line {line_num}: "
                    f"expected {cols}, got {values.size}"
                )
            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"[ERROR] No valid rows found in {txt_path}")

    return n_rows


def txt_to_memmap(txt_path, cols, mmap_path, expected_rows=None, dtype=np.float32, overwrite_cache=False):
    meta_path = mmap_path + ".shape"

    if overwrite_cache:
        safe_remove(mmap_path)
        safe_remove(meta_path)

    if os.path.exists(mmap_path) and os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            shape_str = f.read().strip()
        n_rows, n_cols = map(int, shape_str.split(","))

        if n_cols != cols:
            raise ValueError(
                f"[ERROR] Existing raw memmap cache for {txt_path} has cols={n_cols}, expected {cols}"
            )

        arr = np.memmap(mmap_path, mode="r", dtype=dtype, shape=(n_rows, n_cols))
        return arr, n_rows

    print(f"[INFO] Counting rows / validating columns: {txt_path}")
    actual_rows = count_rows_and_validate_cols(txt_path, cols, dtype=dtype)
    print(f"[INFO] Detected shape: ({actual_rows}, {cols})")

    if expected_rows is not None and actual_rows != expected_rows:
        raise ValueError(
            f"[ERROR] Row mismatch for {txt_path}: expected {expected_rows}, got {actual_rows}"
        )

    print(f"[INFO] Creating raw memmap: {mmap_path}")
    arr = np.memmap(mmap_path, mode="w+", dtype=dtype, shape=(actual_rows, cols))

    row_idx = 0
    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            values = np.fromstring(line, sep=" ", dtype=dtype)
            if values.size != cols:
                raise ValueError(
                    f"[ERROR] Column mismatch in {txt_path} at line {line_num}: "
                    f"expected {cols}, got {values.size}"
                )

            arr[row_idx] = values
            row_idx += 1

    arr.flush()

    with open(meta_path, "w") as f:
        f.write(f"{actual_rows},{cols}")

    return np.memmap(mmap_path, mode="r", dtype=dtype, shape=(actual_rows, cols)), actual_rows


def load_normalized_cache(txt_path, cols, cache_dir, expected_rows=None, overwrite_cache=False):
    """
    Returns:
        normalized_embeddings_memmap, actual_rows
    """
    ensure_dir(cache_dir)

    prefix = cache_prefix(cache_dir, txt_path)
    raw_mmap_path = prefix + ".mmap"
    norm_npy_path = prefix + "_normalized.npy"
    norm_shape_path = prefix + "_normalized.shape"

    if overwrite_cache:
        safe_remove(raw_mmap_path)
        safe_remove(raw_mmap_path + ".shape")
        safe_remove(norm_npy_path)
        safe_remove(norm_shape_path)

    if os.path.exists(norm_npy_path) and os.path.exists(norm_shape_path):
        with open(norm_shape_path, "r") as f:
            shape_str = f.read().strip()
        actual_rows, actual_cols = map(int, shape_str.split(","))

        if actual_cols != cols:
            raise ValueError(
                f"[ERROR] Cached normalized array for {txt_path} has cols={actual_cols}, expected {cols}"
            )

        if expected_rows is not None and actual_rows != expected_rows:
            raise ValueError(
                f"[ERROR] Cached normalized array for {txt_path} has rows={actual_rows}, expected {expected_rows}"
            )

        print(f"[INFO] Reusing normalized cache: {norm_npy_path}")
        emb = np.load(norm_npy_path, mmap_mode="r")
        return emb, actual_rows

    raw_memmap, actual_rows = txt_to_memmap(
        txt_path=txt_path,
        cols=cols,
        mmap_path=raw_mmap_path,
        expected_rows=expected_rows,
        dtype=np.float32,
        overwrite_cache=overwrite_cache
    )

    print(f"[INFO] Normalizing embeddings: {txt_path}")
    emb = np.array(raw_memmap, dtype=np.float32, copy=True)
    faiss.normalize_L2(emb)

    np.save(norm_npy_path, emb)
    with open(norm_shape_path, "w") as f:
        f.write(f"{actual_rows},{cols}")

    print(f"[INFO] Saved normalized cache: {norm_npy_path}")

    del emb
    emb = np.load(norm_npy_path, mmap_mode="r")
    return emb, actual_rows


# ============================================================
# FAISS search helper
# ============================================================
def batched_extract_kth_distances(index, queries, k_values, batch_size=2048, same_dataset=False):
    """
    Extract only the kth-neighbor distances needed for the requested k values.

    If same_dataset=True:
        search with max_k + 1 and remove the self-match (index == query row id).
    If same_dataset=False:
        normal cross-dataset search.
    """
    k_values = sorted(k_values)
    max_k = max(k_values)
    n = queries.shape[0]
    out = np.empty((n, len(k_values)), dtype=np.float32)

    if same_dataset:
        search_k = max_k + 1
    else:
        search_k = max_k

    col_indices = [k - 1 for k in k_values]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        q = np.asarray(queries[start:end], dtype=np.float32)
        D, I = index.search(q, search_k)

        if same_dataset:
            for local_idx in range(end - start):
                global_idx = start + local_idx
                keep_mask = (I[local_idx] != global_idx)
                filtered_D = D[local_idx][keep_mask]

                if filtered_D.shape[0] < max_k:
                    raise ValueError(
                        f"[ERROR] Could not obtain {max_k} non-self neighbors for row {global_idx}"
                    )

                out[global_idx] = filtered_D[col_indices]
        else:
            out[start:end] = D[:, col_indices]

        if ((start // batch_size) + 1) % 20 == 0 or end == n:
            print(f"[INFO] Search progress: {end}/{n}")

    return out


# ============================================================
# Per-model preparation:
# build train index once, extract all needed kth distances once
# ============================================================
def prepare_model_data(
    model_name,
    train_path,
    valid_path,
    test_path,
    dim,
    k_values,
    cache_dir,
    search_batch_size=2048,
    overwrite_cache=False
):
    print("\n" + "#" * 90)
    print(f"[INFO] PREPARING MODEL: {model_name}")
    print("#" * 90)

    train_emb, n_train = load_normalized_cache(
        txt_path=train_path,
        cols=dim,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache
    )
    valid_emb, n_valid = load_normalized_cache(
        txt_path=valid_path,
        cols=dim,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache
    )
    test_emb, n_test = load_normalized_cache(
        txt_path=test_path,
        cols=dim,
        cache_dir=cache_dir,
        overwrite_cache=overwrite_cache
    )

    print(f"[INFO] Train shape: ({n_train}, {dim})")
    print(f"[INFO] Valid shape: ({n_valid}, {dim})")
    print(f"[INFO] Test  shape: ({n_test}, {dim})")

    max_k = max(k_values)
    if max_k > n_train:
        raise ValueError(
            f"[ERROR] Largest k={max_k} is greater than number of train samples={n_train}"
        )

    print(f"[INFO] Building FAISS IndexFlatL2 for {model_name}")
    index = faiss.IndexFlatL2(dim)
    index.add(np.asarray(train_emb, dtype=np.float32))

    # In this OOD workflow validation/test are different from train,
    # so same_dataset=False is the correct setting.
    print(f"[INFO] Extracting validation kth-distances for {model_name}")
    valid_kth = batched_extract_kth_distances(
        index=index,
        queries=valid_emb,
        k_values=k_values,
        batch_size=search_batch_size,
        same_dataset=False
    )

    print(f"[INFO] Extracting test kth-distances for {model_name}")
    test_kth = batched_extract_kth_distances(
        index=index,
        queries=test_emb,
        k_values=k_values,
        batch_size=search_batch_size,
        same_dataset=False
    )

    del train_emb, valid_emb, test_emb, index

    return {
        "model_name": model_name,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "valid_kth": valid_kth,
        "test_kth": test_kth,
    }


# ============================================================
# Save helpers
# ============================================================
def save_ood_indices_csv(csv_path, indices):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ood_index"])
        for idx in indices:
            writer.writerow([int(idx)])


def save_per_k_ood_summary(summary_csv_path, rows):
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "k",
            "threshold",
            "num_train",
            "num_valid",
            "num_test",
            "num_ood",
            "ood_fraction",
            "pickle_path",
            "ood_indices_csv_path",
        ])
        for row in rows:
            writer.writerow([
                row["model"],
                row["k"],
                row["threshold"],
                row["num_train"],
                row["num_valid"],
                row["num_test"],
                row["num_ood"],
                row["ood_fraction"],
                row["pickle_path"],
                row["ood_indices_csv_path"],
            ])


def save_per_k_jaccard_summary(jaccard_csv_path, model_ood_sets):
    models = list(model_ood_sets.keys())

    with open(jaccard_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_a",
            "model_b",
            "size_a",
            "size_b",
            "intersection_size",
            "union_size",
            "jaccard_similarity",
        ])

        for model_a, model_b in itertools.combinations(models, 2):
            set_a = model_ood_sets[model_a]
            set_b = model_ood_sets[model_b]

            inter = len(set_a & set_b)
            union = len(set_a | set_b)
            jac = jaccard_set_similarity(set_a, set_b)

            writer.writerow([
                model_a,
                model_b,
                len(set_a),
                len(set_b),
                inter,
                union,
                jac,
            ])


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    # 7B
    parser.add_argument(
        "--train_7b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/7B/lora_embeddings_train_7B.txt"
    )
    parser.add_argument(
        "--valid_7b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/7B/7B_lora_embeddings_valid.txt"
    )
    parser.add_argument(
        "--test_7b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/PAWS/7B_lora_embeddings_train_PAWS.txt"
    )

    # 13B
    parser.add_argument(
        "--train_13b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/13B/lora_embeddings_train_13B.txt"
    )
    parser.add_argument(
        "--valid_13b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/13B/13B_lora_embeddings_valid.txt"
    )
    parser.add_argument(
        "--test_13b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/PAWS/13B_lora_embeddings_train_PAWS.txt"
    )

    # 70B
    parser.add_argument(
        "--train_70b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/70B/lora_embeddings_train_70B.txt"
    )
    parser.add_argument(
        "--valid_70b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/fine-tune/embeddings/70B/70B_lora_embeddings_valid.txt"
    )
    parser.add_argument(
        "--test_70b",
        type=str,
        default="/home/gpuuser2/gpuuser2_a/rohit/llama-2-13B-hf/PI/PAWS/70B_lora_embeddings_train_PAWS.txt"
    )

    parser.add_argument("--dim_7b", type=int, default=4096)
    parser.add_argument("--dim_13b", type=int, default=5120)
    parser.add_argument("--dim_70b", type=int, default=8192)

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500, 800, 1000]
    )
    parser.add_argument("--percent", type=float, default=95.0)

    parser.add_argument("--cache_dir", type=str, default="./embedding_cache_ood_llama")
    parser.add_argument("--output_root", type=str, default="./ood_llama_outputs")

    parser.add_argument("--search_batch_size", type=int, default=2048)
    parser.add_argument("--overwrite_cache", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.cache_dir)
    ensure_dir(args.output_root)
    ensure_dir(os.path.join(args.output_root, "summary"))

    k_values = sorted(set(args.k_values))

    model_specs = {
        "Llama-2-7B": {
            "train": args.train_7b,
            "valid": args.valid_7b,
            "test": args.test_7b,
            "dim": args.dim_7b,
        },
        "Llama-2-13B": {
            "train": args.train_13b,
            "valid": args.valid_13b,
            "test": args.test_13b,
            "dim": args.dim_13b,
        },
        "Llama-2-70B": {
            "train": args.train_70b,
            "valid": args.valid_70b,
            "test": args.test_70b,
            "dim": args.dim_70b,
        },
    }

    # --------------------------------------------------------
    # Prepare each model once:
    # build index once, search valid/test once up to max(k)
    # --------------------------------------------------------
    prepared = {}

    for model_name, spec in model_specs.items():
        prepared[model_name] = prepare_model_data(
            model_name=model_name,
            train_path=spec["train"],
            valid_path=spec["valid"],
            test_path=spec["test"],
            dim=spec["dim"],
            k_values=k_values,
            cache_dir=args.cache_dir,
            search_batch_size=args.search_batch_size,
            overwrite_cache=args.overwrite_cache
        )

    # --------------------------------------------------------
    # Process one k at a time and save immediately
    # --------------------------------------------------------
    for j, k in enumerate(k_values):
        print("\n" + "=" * 100)
        print(f"[INFO] PROCESSING k = {k}")
        print("=" * 100)

        per_k_rows = []
        per_k_ood_sets = {}

        for model_name in model_specs.keys():
            data = prepared[model_name]

            kth_values_val = data["valid_kth"][:, j]
            threshold = select_threshold_func(kth_values_val, args.percent)

            kth_values_test = data["test_kth"][:, j]
            ood_mask = (kth_values_test > threshold)
            ood_idx = np.where(ood_mask)[0].astype(np.int64)
            ood_set = set(ood_idx.tolist())

            per_k_ood_sets[model_name] = ood_set

            save_dir = os.path.join(args.output_root, model_name, f"k_{k}", "QQP_to_PAWS")
            ensure_dir(save_dir)

            pickle_path = os.path.join(save_dir, f"{model_name}.pkl")
            csv_indices_path = os.path.join(save_dir, f"{model_name}_ood_indices.csv")

            with open(pickle_path, "wb") as f:
                pickle.dump(ood_set, f)

            save_ood_indices_csv(csv_indices_path, sorted(ood_set))

            row = {
                "model": model_name,
                "k": k,
                "threshold": threshold,
                "num_train": data["n_train"],
                "num_valid": data["n_valid"],
                "num_test": data["n_test"],
                "num_ood": len(ood_set),
                "ood_fraction": len(ood_set) / data["n_test"] if data["n_test"] > 0 else 0.0,
                "pickle_path": pickle_path,
                "ood_indices_csv_path": csv_indices_path,
            }
            per_k_rows.append(row)

            print(
                f"[INFO] {model_name} | k={k} | threshold={threshold:.6f} | "
                f"OOD count={len(ood_set)}"
            )

        # Save per-k summary CSV immediately
        summary_csv_path = os.path.join(args.output_root, "summary", f"ood_summary_k{k}.csv")
        save_per_k_ood_summary(summary_csv_path, per_k_rows)

        # Save per-k pairwise Jaccard CSV immediately
        jaccard_csv_path = os.path.join(args.output_root, "summary", f"ood_jaccard_k{k}.csv")
        save_per_k_jaccard_summary(jaccard_csv_path, per_k_ood_sets)

        print(f"[INFO] Saved summary CSV   : {summary_csv_path}")
        print(f"[INFO] Saved Jaccard CSV   : {jaccard_csv_path}")
        print("-" * 100)

    print("\nDone.")


if __name__ == "__main__":
    main()
