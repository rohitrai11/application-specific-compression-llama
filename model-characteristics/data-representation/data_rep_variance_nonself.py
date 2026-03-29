import os
import csv
import math
import argparse
import itertools
import numpy as np
import faiss


# ============================================================
# Utility helpers
# ============================================================
def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Step 1: Validate embedding text files and build caches
# ============================================================
def count_rows_and_validate_cols(txt_path, cols, dtype=np.float32):
    """
    First pass over the text file:
    - count non-empty rows
    - validate each row has exactly `cols` values
    """
    n_rows = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            values = np.fromstring(line, sep=" ", dtype=dtype)

            if values.size != cols:
                raise ValueError(
                    f"[ERROR] Column mismatch in file {txt_path} at line {line_num}: "
                    f"expected {cols} values, got {values.size}"
                )

            n_rows += 1

    if n_rows == 0:
        raise ValueError(f"[ERROR] No valid embedding rows found in {txt_path}")

    return n_rows


def txt_to_memmap(txt_path, cols, mmap_path, expected_rows=None, dtype=np.float32, overwrite_cache=False):
    """
    Convert a huge whitespace-separated embedding .txt file into a memmap cache.

    Returns:
        raw_memmap, actual_rows
    """
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
                f"[ERROR] Existing memmap cache has cols={n_cols}, expected {cols}. "
                f"Delete cache or use --overwrite_cache."
            )

        arr = np.memmap(mmap_path, mode="r", dtype=dtype, shape=(n_rows, n_cols))
        return arr, n_rows

    print(f"[INFO] Counting rows and validating columns for: {txt_path}")
    actual_rows = count_rows_and_validate_cols(txt_path, cols, dtype=dtype)
    print(f"[INFO] Actual detected shape: ({actual_rows}, {cols})")

    if expected_rows is not None and actual_rows != expected_rows:
        raise ValueError(
            f"[ERROR] Row mismatch for {txt_path}: expected {expected_rows}, got {actual_rows}. "
            f"Either fix --rows or omit it for auto-detection."
        )

    print(f"[INFO] Creating memmap cache for: {txt_path}")
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
                    f"[ERROR] Column mismatch in file {txt_path} at line {line_num}: "
                    f"expected {cols} values, got {values.size}"
                )

            arr[row_idx] = values
            row_idx += 1

    arr.flush()

    with open(meta_path, "w") as f:
        f.write(f"{actual_rows},{cols}")

    return np.memmap(mmap_path, mode="r", dtype=dtype, shape=(actual_rows, cols)), actual_rows


def load_embedding_with_cache(txt_path, cols, cache_dir, expected_rows=None, overwrite_cache=False):
    """
    1. txt -> memmap cache
    2. normalize embeddings
    3. save normalized .npy cache

    Returns:
        normalized_embeddings_memmap, actual_rows
    """
    ensure_dir(cache_dir)

    base_name = os.path.splitext(os.path.basename(txt_path))[0]

    mmap_path = os.path.join(cache_dir, f"{base_name}.mmap")
    norm_npy_path = os.path.join(cache_dir, f"{base_name}_normalized.npy")
    norm_shape_path = os.path.join(cache_dir, f"{base_name}_normalized.shape")

    if overwrite_cache:
        safe_remove(mmap_path)
        safe_remove(mmap_path + ".shape")
        safe_remove(norm_npy_path)
        safe_remove(norm_shape_path)

    if os.path.exists(norm_npy_path) and os.path.exists(norm_shape_path):
        with open(norm_shape_path, "r") as f:
            shape_str = f.read().strip()
        actual_rows, actual_cols = map(int, shape_str.split(","))

        if actual_cols != cols:
            raise ValueError(
                f"[ERROR] Cached normalized array has cols={actual_cols}, expected {cols}. "
                f"Delete cache or use --overwrite_cache."
            )

        if expected_rows is not None and actual_rows != expected_rows:
            raise ValueError(
                f"[ERROR] Cached normalized row count {actual_rows} does not match expected_rows={expected_rows}"
            )

        print(f"[INFO] Loading normalized cache: {norm_npy_path}")
        emb = np.load(norm_npy_path, mmap_mode="r")
        return emb, actual_rows

    raw_memmap, actual_rows = txt_to_memmap(
        txt_path=txt_path,
        cols=cols,
        mmap_path=mmap_path,
        expected_rows=expected_rows,
        dtype=np.float32,
        overwrite_cache=overwrite_cache,
    )

    print(f"[INFO] Normalizing: {txt_path}")
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
# Step 2: Build neighbor cache WITHOUT self-neighbors
# ============================================================
def build_neighbor_cache_without_self(
    embeddings,
    neighbor_cache_path,
    max_k,
    batch_size=2048,
    overwrite_cache=False
):
    """
    Build FAISS exact L2 index and cache top-max_k neighbors excluding self.

    We search with (max_k + 1) because self is usually included.
    Then we remove self and keep the first max_k non-self neighbors.

    Neighbor cache is stored as .npy with shape: (num_samples, max_k), dtype=int32
    """
    if overwrite_cache:
        safe_remove(neighbor_cache_path)

    if os.path.exists(neighbor_cache_path):
        arr = np.load(neighbor_cache_path, mmap_mode="r")
        if arr.shape[1] != max_k:
            raise ValueError(
                f"[ERROR] Existing neighbor cache {neighbor_cache_path} has second dimension "
                f"{arr.shape[1]}, expected {max_k}. Delete cache or use --overwrite_cache."
            )
        print(f"[INFO] Reusing neighbor cache: {neighbor_cache_path}")
        return neighbor_cache_path

    num_rows, dim = embeddings.shape

    if max_k >= num_rows:
        raise ValueError(
            f"[ERROR] max_k={max_k} must be smaller than num_rows={num_rows}"
        )

    print(f"[INFO] Building FAISS index for embeddings of shape {embeddings.shape}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"[INFO] Creating non-self neighbor cache: {neighbor_cache_path}")
    neigh_memmap = np.lib.format.open_memmap(
        neighbor_cache_path,
        mode="w+",
        dtype=np.int32,
        shape=(num_rows, max_k)
    )

    search_k = max_k + 1

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        queries = embeddings[start:end]

        _, I = index.search(queries, search_k)

        for local_idx in range(end - start):
            global_idx = start + local_idx
            row = I[local_idx]

            filtered = row[row != global_idx]

            if filtered.shape[0] < max_k:
                raise ValueError(
                    f"[ERROR] Could not obtain {max_k} non-self neighbors for sample {global_idx}. "
                    f"Returned row length after removing self: {filtered.shape[0]}"
                )

            neigh_memmap[global_idx] = filtered[:max_k].astype(np.int32)

        if ((start // batch_size) + 1) % 10 == 0 or end == num_rows:
            print(f"[INFO] Neighbor cache progress: {end}/{num_rows}")

    del neigh_memmap
    print(f"[INFO] Saved non-self neighbor cache: {neighbor_cache_path}")
    return neighbor_cache_path


# ============================================================
# Step 3: Jaccard computation for multiple k values
# ============================================================
def jaccard_scores_from_slices(neigh_a, neigh_b, k):
    """
    Compute per-row Jaccard using only first k neighbors from each row.
    neigh_a, neigh_b shape: (chunk_size, >=k)

    Since each row has exactly k unique non-self neighbors:
        union_size = 2*k - intersection_size
    """
    chunk_size = neigh_a.shape[0]
    scores = np.empty(chunk_size, dtype=np.float32)

    for i in range(chunk_size):
        set_a = set(neigh_a[i, :k].tolist())
        set_b = set(neigh_b[i, :k].tolist())
        inter = len(set_a & set_b)
        scores[i] = inter / float(2 * k - inter)

    return scores


def summarize_scores_from_running_stats(count, sum_scores, sumsq_scores, min_score, max_score):
    """
    Sample variance with ddof=1, same logic as statistics.variance / numpy var(ddof=1).
    """
    if count == 0:
        raise ValueError("[ERROR] No scores to summarize.")

    mean_score = sum_scores / count

    if count > 1:
        variance = (sumsq_scores - (sum_scores * sum_scores) / count) / (count - 1)
        variance = max(0.0, variance)
        std_dev = math.sqrt(variance)
    else:
        variance = 0.0
        std_dev = 0.0

    return {
        "num_samples": int(count),
        "mean_jaccard": float(mean_score),
        "variance_jaccard": float(variance),
        "std_jaccard": float(std_dev),
        "min_jaccard": float(min_score),
        "max_jaccard": float(max_score),
    }


def compute_and_save_for_k(
    k,
    model_pairs,
    neighbor_cache_paths,
    output_dir,
    score_chunk_size=10000,
    overwrite_results=False
):
    """
    For one k:
    - compute pairwise Jaccard scores
    - write detailed CSV incrementally while computing
    - write summary CSV immediately after that k finishes
    """
    ensure_dir(output_dir)

    detailed_csv = os.path.join(output_dir, f"pairwise_jaccard_detailed_k{k}_nonself.csv")
    summary_csv = os.path.join(output_dir, f"pairwise_jaccard_summary_k{k}_nonself.csv")

    if overwrite_results:
        safe_remove(detailed_csv)
        safe_remove(summary_csv)

    # Write header immediately
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "model_a", "model_b", "jaccard_score"])

    summary_rows = []

    for model_a, model_b in model_pairs:
        print("\n" + "=" * 80)
        print(f"[INFO] Computing Jaccard for k={k}: {model_a} vs {model_b}")
        print("=" * 80)

        neigh_a = np.load(neighbor_cache_paths[model_a], mmap_mode="r")
        neigh_b = np.load(neighbor_cache_paths[model_b], mmap_mode="r")

        if neigh_a.shape != neigh_b.shape:
            raise ValueError(
                f"[ERROR] Neighbor cache shape mismatch: {model_a} {neigh_a.shape} vs {model_b} {neigh_b.shape}"
            )

        num_rows = neigh_a.shape[0]

        count = 0
        sum_scores = 0.0
        sumsq_scores = 0.0
        min_score = float("inf")
        max_score = float("-inf")

        with open(detailed_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for start in range(0, num_rows, score_chunk_size):
                end = min(start + score_chunk_size, num_rows)

                scores = jaccard_scores_from_slices(
                    neigh_a[start:end],
                    neigh_b[start:end],
                    k
                )

                rows_to_write = [
                    (start + i, model_a, model_b, float(scores[i]))
                    for i in range(len(scores))
                ]
                writer.writerows(rows_to_write)

                chunk_count = len(scores)
                chunk_sum = float(scores.sum(dtype=np.float64))
                chunk_sumsq = float(np.square(scores, dtype=np.float64).sum(dtype=np.float64))
                chunk_min = float(scores.min())
                chunk_max = float(scores.max())

                count += chunk_count
                sum_scores += chunk_sum
                sumsq_scores += chunk_sumsq
                min_score = min(min_score, chunk_min)
                max_score = max(max_score, chunk_max)

                if ((start // score_chunk_size) + 1) % 20 == 0 or end == num_rows:
                    print(f"[INFO] k={k}, {model_a} vs {model_b}: {end}/{num_rows}")

        stats = summarize_scores_from_running_stats(
            count=count,
            sum_scores=sum_scores,
            sumsq_scores=sumsq_scores,
            min_score=min_score,
            max_score=max_score,
        )

        summary_row = {
            "model_a": model_a,
            "model_b": model_b,
            **stats
        }
        summary_rows.append(summary_row)

        print(
            f"[INFO] k={k}, {model_a} vs {model_b} | "
            f"mean={stats['mean_jaccard']:.6f}, "
            f"var={stats['variance_jaccard']:.6f}, "
            f"std={stats['std_jaccard']:.6f}, "
            f"min={stats['min_jaccard']:.6f}, "
            f"max={stats['max_jaccard']:.6f}"
        )

    # Save summary as soon as this k is done
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_a",
            "model_b",
            "num_samples",
            "mean_jaccard",
            "variance_jaccard",
            "std_jaccard",
            "min_jaccard",
            "max_jaccard",
        ])
        for row in summary_rows:
            writer.writerow([
                row["model_a"],
                row["model_b"],
                row["num_samples"],
                row["mean_jaccard"],
                row["variance_jaccard"],
                row["std_jaccard"],
                row["min_jaccard"],
                row["max_jaccard"],
            ])

    print("\n" + "-" * 80)
    print(f"[INFO] Finished k={k}")
    print(f"[INFO] Detailed CSV saved: {detailed_csv}")
    print(f"[INFO] Summary  CSV saved: {summary_csv}")
    print("-" * 80)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pairwise Jaccard comparison of LLaMA embeddings using non-self nearest neighbors."
    )

    parser.add_argument("--input_file_7b", type=str, required=True,
                        help="Path to base_embeddings_train_7B.txt")
    parser.add_argument("--input_file_13b", type=str, required=True,
                        help="Path to base_embeddings_train_13B.txt")
    parser.add_argument("--input_file_70b", type=str, required=True,
                        help="Path to base_embeddings_train_70B.txt")

    parser.add_argument("--rows", type=int, default=None,
                        help="Expected number of rows. Omit for auto-detection.")
    parser.add_argument("--dim_7b", type=int, default=4096,
                        help="Embedding dimension for 7B")
    parser.add_argument("--dim_13b", type=int, default=5120,
                        help="Embedding dimension for 13B")
    parser.add_argument("--dim_70b", type=int, default=8192,
                        help="Embedding dimension for 70B")

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500, 800, 1000],
        help="List of k values to evaluate"
    )

    parser.add_argument("--cache_dir", type=str, default="./embedding_cache",
                        help="Directory for memmap / normalized caches / neighbor caches")
    parser.add_argument("--output_dir", type=str, default="./results_nonself",
                        help="Directory for result CSV files")

    parser.add_argument("--neighbor_batch_size", type=int, default=2048,
                        help="Batch size while running FAISS search for neighbor-cache creation")
    parser.add_argument("--score_chunk_size", type=int, default=10000,
                        help="Chunk size while computing Jaccard and writing CSVs")

    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Delete and rebuild embedding caches and neighbor caches")
    parser.add_argument("--overwrite_results", action="store_true",
                        help="Overwrite per-k CSV files if they already exist")

    args = parser.parse_args()

    ensure_dir(args.cache_dir)
    ensure_dir(args.output_dir)

    k_values = sorted(set(args.k_values))
    max_k = max(k_values)

    model_specs = {
        "LLaMA-7B":  (args.input_file_7b, args.dim_7b),
        "LLaMA-13B": (args.input_file_13b, args.dim_13b),
        "LLaMA-70B": (args.input_file_70b, args.dim_70b),
    }

    actual_row_counts = {}
    neighbor_cache_paths = {}

    # --------------------------------------------------------
    # Build normalized embedding cache + neighbor cache per model
    # --------------------------------------------------------
    for model_name, (file_path, dim) in model_specs.items():
        print("\n" + "#" * 80)
        print(f"[INFO] Processing {model_name}")
        print(f"[INFO] File: {file_path}")
        print(f"[INFO] Expected columns: {dim}")
        if args.rows is None:
            print("[INFO] Expected rows: auto-detect")
        else:
            print(f"[INFO] Expected rows: {args.rows}")
        print(f"[INFO] Max k requested: {max_k}")
        print("#" * 80)

        emb, actual_rows = load_embedding_with_cache(
            txt_path=file_path,
            cols=dim,
            cache_dir=args.cache_dir,
            expected_rows=args.rows,
            overwrite_cache=args.overwrite_cache
        )

        print(f"[INFO] Loaded normalized embeddings for {model_name}: {emb.shape}")
        actual_row_counts[model_name] = actual_rows

        neighbor_cache_path = os.path.join(
            args.cache_dir,
            f"{model_name.replace('-', '_')}_neighbors_nonself_top{max_k}.npy"
        )

        build_neighbor_cache_without_self(
            embeddings=emb,
            neighbor_cache_path=neighbor_cache_path,
            max_k=max_k,
            batch_size=args.neighbor_batch_size,
            overwrite_cache=args.overwrite_cache
        )

        neighbor_cache_paths[model_name] = neighbor_cache_path

        # release embedding reference before next model
        del emb

    # --------------------------------------------------------
    # Validate row consistency across models
    # --------------------------------------------------------
    unique_row_counts = set(actual_row_counts.values())
    if len(unique_row_counts) != 1:
        raise ValueError(
            f"[ERROR] Row count mismatch across models: {actual_row_counts}"
        )

    common_rows = next(iter(unique_row_counts))
    if max_k >= common_rows:
        raise ValueError(
            f"[ERROR] Largest requested k={max_k} must be smaller than num_rows={common_rows}"
        )

    print("\n" + "-" * 80)
    print(f"[INFO] All models have consistent row count: {common_rows}")
    print(f"[INFO] k values to evaluate: {k_values}")
    print("-" * 80)

    # --------------------------------------------------------
    # Pair definitions
    # --------------------------------------------------------
    model_pairs = list(itertools.combinations(model_specs.keys(), 2))

    # --------------------------------------------------------
    # Compute and save results for each k immediately
    # --------------------------------------------------------
    for k in k_values:
        compute_and_save_for_k(
            k=k,
            model_pairs=model_pairs,
            neighbor_cache_paths=neighbor_cache_paths,
            output_dir=args.output_dir,
            score_chunk_size=args.score_chunk_size,
            overwrite_results=args.overwrite_results
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
