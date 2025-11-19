# ============================================================
#  DeBERTa-v3-small Binary Classifier (Distillation)
#  Input: Profanity_LLM.csv
#  Output: attack_flag classifier
# ============================================================

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import evaluate
from sklearn.model_selection import train_test_split

import torch
torch.backends.mps.is_available = lambda: False
torch.backends.mps.is_built = lambda: False

# ============================================================
# 1. Load and prepare dataset
# ============================================================

df = pd.read_csv("Profanity_LLM.csv")

# Convert three boolean columns into single binary target
df["attack_flag"] = (
    df["attack_quote"].astype(int) |
    df["attack_mention"].astype(int) |
    df["attack_named_person"].astype(int)
)

# Build the training text input
def merge_fields(row):
    parts = []
    if isinstance(row["text"], str) and row["text"].strip():
        parts.append(f"Post: {row['text']}")
    if isinstance(row.get("quoted_text", ""), str) and row["quoted_text"].strip():
        parts.append(f"Quoted: {row['quoted_text']}")
    if isinstance(row.get("mentions_handle", ""), str) and row["mentions_handle"].strip():
        parts.append(f"Mentions: {row['mentions_handle']}")
    if isinstance(row.get("tags", ""), str) and row["tags"].strip():
        parts.append(f"Tags: {row['tags']}")

    return "\n".join(parts) if parts else "[EMPTY_POST]"

df["input_text"] = df.apply(merge_fields, axis=1)

# Keep only necessary columns
df = df[["input_text", "attack_flag"]]

# Filter out empty rows
df = df[df["input_text"].notna()]

print("Dataset size:", len(df))
print(df.head())

# ============================================================
# 2. Train/validation split
# ============================================================

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ============================================================
# 3. Load tokenizer and model
# ============================================================

model_name = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,                          # binary classification → 1 sigmoid output
    problem_type="multi_label_classification"
)

# ============================================================
# 4. Tokenization function
# ============================================================

def tokenize(batch):
    tokens = tokenizer(
        batch["input_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokens["labels"] = [[float(x)] for x in batch["attack_flag"]]
    return tokens

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["input_text", "__index_level_0__"])
val_dataset = val_dataset.remove_columns(["input_text", "__index_level_0__"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

for x in train_dataset[:5]["labels"]:
    print(x, type(x))

# ============================================================
# 5. Evaluation metrics
# ============================================================

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    acc = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    pre = precision.compute(predictions=preds, references=labels)["precision"]
    rec = recall.compute(predictions=preds, references=labels)["recall"]
    f1v = f1.compute(predictions=preds, references=labels)["f1"]

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1v
    }

# ============================================================
# 6. Training configuration
# ============================================================

training_args = TrainingArguments(
    output_dir="./attack_classifier_deberta",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    num_train_epochs=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=False,   # Enable if GPU supports FP16
    no_cuda=True,
)

# ============================================================
# 7. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ============================================================
# 8. Train!
# ============================================================

trainer.train()

# ============================================================
# 9. Save fine-tuned model
# ============================================================

trainer.save_model("./attack_classifier_deberta_final")
tokenizer.save_pretrained("./attack_classifier_deberta_final")

print("🎉 Training complete! Model saved to ./attack_classifier_deberta_final")


"""
{'loss': 0.5534, 'grad_norm': 4.7030839920043945, 'learning_rate': 4.6936274509803925e-05, 'epoch': 0.25}                                                        
{'loss': 0.4556, 'grad_norm': 2.439638614654541, 'learning_rate': 4.387254901960784e-05, 'epoch': 0.49}                                                          
{'loss': 0.4129, 'grad_norm': 2.425090789794922, 'learning_rate': 4.0808823529411765e-05, 'epoch': 0.74}                                                         
{'loss': 0.4296, 'grad_norm': 4.1031575202941895, 'learning_rate': 3.774509803921569e-05, 'epoch': 0.98}                                                         
{'eval_loss': 0.3573993742465973, 'eval_accuracy': 0.8476658476658476, 'eval_precision': 0.824468085106383, 'eval_recall': 0.9489795918367347, 'eval_f1': 0.8823529411764706, 'eval_runtime': 35.1954, 'eval_samples_per_second': 23.128, 'eval_steps_per_second': 1.449, 'epoch': 1.0}                                           
{'loss': 0.3409, 'grad_norm': 4.279551029205322, 'learning_rate': 3.468137254901961e-05, 'epoch': 1.23}                                                          
{'loss': 0.2921, 'grad_norm': 7.918822288513184, 'learning_rate': 3.161764705882353e-05, 'epoch': 1.47}                                                          
{'loss': 0.3306, 'grad_norm': 1.1836107969284058, 'learning_rate': 2.855392156862745e-05, 'epoch': 1.72}                                                         
{'loss': 0.2647, 'grad_norm': 8.383955001831055, 'learning_rate': 2.5490196078431373e-05, 'epoch': 1.96}                                                         
{'eval_loss': 0.3318573534488678, 'eval_accuracy': 0.8734643734643734, 'eval_precision': 0.8924949290060852, 'eval_recall': 0.8979591836734694, 'eval_f1': 0.8952187182095626, 'eval_runtime': 34.2725, 'eval_samples_per_second': 23.751, 'eval_steps_per_second': 1.488, 'epoch': 2.0}                                          
{'loss': 0.1867, 'grad_norm': 7.751635551452637, 'learning_rate': 2.2426470588235296e-05, 'epoch': 2.21}                                                         
{'loss': 0.1671, 'grad_norm': 6.533549785614014, 'learning_rate': 1.936274509803922e-05, 'epoch': 2.45}                                                          
{'loss': 0.2111, 'grad_norm': 6.200825214385986, 'learning_rate': 1.6299019607843138e-05, 'epoch': 2.7}                                                          
{'loss': 0.2065, 'grad_norm': 18.59579086303711, 'learning_rate': 1.323529411764706e-05, 'epoch': 2.94}                                                          
{'eval_loss': 0.4736766815185547, 'eval_accuracy': 0.8574938574938575, 'eval_precision': 0.8327402135231317, 'eval_recall': 0.9551020408163265, 'eval_f1': 0.8897338403041825, 'eval_runtime': 35.1478, 'eval_samples_per_second': 23.159, 'eval_steps_per_second': 1.451, 'epoch': 3.0}                                          
{'loss': 0.1174, 'grad_norm': 5.8887224197387695, 'learning_rate': 1.017156862745098e-05, 'epoch': 3.19}                                                         
{'loss': 0.1123, 'grad_norm': 0.34891074895858765, 'learning_rate': 7.107843137254902e-06, 'epoch': 3.43}                                                        
{'loss': 0.1154, 'grad_norm': 0.9087517857551575, 'learning_rate': 4.044117647058824e-06, 'epoch': 3.68}                                                         
{'loss': 0.0785, 'grad_norm': 0.8971465229988098, 'learning_rate': 9.80392156862745e-07, 'epoch': 3.92}                                                          
{'eval_loss': 0.5505579710006714, 'eval_accuracy': 0.8685503685503686, 'eval_precision': 0.8732943469785575, 'eval_recall': 0.9142857142857143, 'eval_f1': 0.8933200398803589, 'eval_runtime': 35.5356, 'eval_samples_per_second': 22.907, 'eval_steps_per_second': 1.435, 'epoch': 4.0}                                          
{'train_runtime': 2161.1843, 'train_samples_per_second': 6.024, 'train_steps_per_second': 0.378, 'train_loss': 0.26416366255166485, 'epoch': 4.0}    
"""
