# =========================
# 1. Imports
# =========================
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


# =========================
# 2. Config
# =========================
base_model_name = "/home/gpuuser0/gpuuser0_a/rohit/llama-11B-NLI/pretrained-model/model"
lora_model_path = "llama-2-7B-fine-tuned-model/best_model"

BATCH_SIZE = 16
MAX_LENGTH = 512


# =========================
# 3. Load Dataset
# =========================
df_test = pd.read_csv("../dataset/splits/train.csv")
df_test = df_test[df_test["is_duplicate"] != "unknown"].reset_index(drop=True)

def combine_text(row):
    return f"question1: {row['question1']} question2: {row['question2']}"

df_test["text"] = df_test.apply(combine_text, axis=1)

if "id" not in df_test.columns:
    df_test["id"] = range(len(df_test))

texts = df_test["text"].tolist()
print(f"Total samples: {len(texts)}")


# =========================
# 4. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# =========================
# 5. Quantization Config
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# =========================
# 6. Helpers
# =========================
def get_input_device(model):
    return model.get_input_embeddings().weight.device

def generate_embeddings(model, texts, batch_size=16):
    all_embeddings = []

    device = get_input_device(model)

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # For AutoModelForCausalLM, use the final hidden states
        last_hidden = outputs.hidden_states[-1]   # [B, T, H]

        attention_mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)

        embeddings = summed / counts
        all_embeddings.append(embeddings.float().cpu().numpy())

    return np.vstack(all_embeddings)


# =====================================
# 🔹 PART 1: BASE MODEL EMBEDDINGS
# =====================================
print("\nLoading BASE model...")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

base_model.eval()

print("\nGenerating BASE embeddings...")
base_embeddings = generate_embeddings(base_model, texts, BATCH_SIZE)

print("Base embeddings shape:", base_embeddings.shape)

#np.save("base_embeddings.npy", base_embeddings)
#pd.DataFrame(base_embeddings).to_csv("base_embeddings.csv", index=False)
np.savetxt("base_embeddings_train.txt", base_embeddings, fmt="%.6f", delimiter="\t")

# Free memory before loading LoRA model
del base_model
gc.collect()
torch.cuda.empty_cache()


# =====================================
# 🔹 PART 2: LoRA MODEL EMBEDDINGS
# =====================================
print("\nChecking LoRA config...")
peft_config = PeftConfig.from_pretrained(lora_model_path)
print("LoRA task type:", peft_config.task_type)

print("\nLoading BASE model again for LoRA...")
base_model_lora = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

print("Loading LoRA weights...")
lora_model = PeftModel.from_pretrained(base_model_lora, lora_model_path)
lora_model.eval()

print("\nGenerating LoRA embeddings...")
lora_embeddings = generate_embeddings(lora_model, texts, BATCH_SIZE)

print("LoRA embeddings shape:", lora_embeddings.shape)

#np.save("lora_embeddings.npy", lora_embeddings)
#pd.DataFrame(lora_embeddings).to_csv("lora_embeddings.csv", index=False)
np.savetxt("lora_embeddings_train.txt", lora_embeddings, fmt="%.6f", delimiter="\t")

# =========================
# 7. Save Metadata
# =========================
meta_df = df_test[["id", "question1", "question2", "is_duplicate"]]
meta_df.to_csv("metadata.csv", index=False)


# =========================
# 8. Done
# =========================
print("\n✅ All embeddings generated and saved!")
print("Files created:")
print("- base_embeddings.npy")
print("- lora_embeddings.npy")
print("- base_embeddings.csv")
print("- lora_embeddings.csv")
print("- metadata.csv")


base_embeddings = np.loadtxt("base_embeddings_train.txt", delimiter="\t")
lora_embeddings = np.loadtxt("lora_embeddings_train.txt", delimiter="\t")

print("Base embeddings shape:", base_embeddings.shape)
print("LoRA embeddings shape:", lora_embeddings.shape)

'''
import torch
import numpy as np
import time

def forward_llama(args, train_dataloader, model, tokenizer, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    embeddings_list = []

    start_time = time.time()
    print("Start time:", start_time)

    with open(output_path, "w") as f:

        for batch in train_dataloader:

            # Move batch to device
            batch = tuple(t.to(device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }

            with torch.no_grad():
                outputs = model(**inputs)

            # ✅ LLaMA embeddings
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            # Masked mean pooling
            masked = last_hidden * attention_mask.unsqueeze(-1)
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1, keepdim=True)

            batch_embeddings = summed / counts   # (B, D)

            batch_embeddings = batch_embeddings.cpu().numpy()

            # Save incrementally (IMPORTANT for large N)
            for emb in batch_embeddings:
                emb_str = " ".join(f"{x:.6f}" for x in emb)
                f.write(emb_str + "\n")

            embeddings_list.extend(batch_embeddings)

            print(f"Processed samples: {len(embeddings_list)}")

    print("Total embeddings:", len(embeddings_list))
    print("Embedding dimension:", len(embeddings_list[0]))

    return np.array(embeddings_list)
'''
