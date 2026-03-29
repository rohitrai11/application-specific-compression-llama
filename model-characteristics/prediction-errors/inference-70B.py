# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 2. Load Test Data
# =========================
df_test = pd.read_csv("../dataset/splits/test.csv")

# Remove 'unknown'
df_test = df_test[df_test['is_duplicate'] != 'unknown'].reset_index(drop=True)

# Limit (optional)
#df_test = df_test.head(5)


# =========================
# 3. Prompt Formatting
# =========================
def generate_test_prompt(row):
    return f"""
Decide whether the following two questions are duplicates.
Answer with exactly one word: yes or no.

question1: {row["question1"]}
question2: {row["question2"]}
label:""".strip()

df_test["text"] = df_test.apply(generate_test_prompt, axis=1)

# Ground truth
y_true = df_test["is_duplicate"].values


# =========================
# 4. Load Base Model + LoRA
# =========================
base_model_name = "/home/gpuuser0/gpuuser0_a/rohit/llama-11B-NLI/pretrained-model/model"
best_model_path = "llama-2-7B-fine-tuned-model/best_model"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

print("Loading LoRA fine-tuned weights...")
model = PeftModel.from_pretrained(base_model, best_model_path)

model.eval()


# =========================
# 5. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id


# =========================
# 6. Create Pipeline (ONLY ONCE)
# =========================
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2,
    temperature=0.0,
    do_sample=False
)


# =========================
# 7. Prediction Function
# =========================
def predict(test_df):
    predictions = []

    for i in tqdm(range(len(test_df))):
        prompt = test_df.iloc[i]["text"]

        output = pipe(prompt)[0]["generated_text"]

        # Extract answer
        answer = output.split("label:")[-1].strip().lower()

        if "yes" in answer:
            predictions.append("yes")
        elif "no" in answer:
            predictions.append("no")
        else:
            predictions.append("none")

    return predictions


# =========================
# 8. Run Predictions
# =========================
print("Running inference...")
y_pred = predict(df_test)


# =========================
# 9. Save Results
# =========================
'''
results_df = pd.DataFrame({
    "question1": df_test["question1"],
    "question2": df_test["question2"],
    "ground_truth": y_true,
    "prediction": y_pred
})

results_df.to_csv("test_predictions.csv", index=False)

print("Saved predictions to test_predictions.csv")
'''

# =========================
# 9. Save Results (WITH ID)
# =========================

# Try to detect ID column automatically
possible_id_cols = ["id", "qid", "pair_id"]

id_col = None
for col in possible_id_cols:
    if col in df_test.columns:
        id_col = col
        break

# If no ID column found, create one
if id_col is None:
    print("No ID column found. Creating one...")
    df_test["id"] = range(len(df_test))
    id_col = "id"

# Create results dataframe
results_df = pd.DataFrame({
    "id": df_test[id_col],
    "question1": df_test["question1"],
    "question2": df_test["question2"],
    "ground_truth": y_true,
    "prediction": y_pred
})

# Save to CSV
results_df.to_csv("test_predictions.csv", index=False)

print(f"Saved predictions with IDs using column: {id_col}")

# =========================
# 10. Evaluation
# =========================
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Final Test Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
