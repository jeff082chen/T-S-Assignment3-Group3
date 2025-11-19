import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from typing import Any, Dict
import warnings
warnings.filterwarnings("ignore", message="`num_beams` is set to None")

# -----------------------------
# Model imports
# -----------------------------
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


# ================================================================
# Utility
# ================================================================

def norm(v: Any) -> str:
    """Normalize cell value to clean string (no NaN)."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


# ================================================================
# Step A: Translation (small100)
# ================================================================

def load_translation_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}_tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}_model")
    tokenizer.tgt_lang = "en"

    gen_cfg = model.generation_config
    gen_cfg.num_beams = 5
    gen_cfg.max_length = 256
    gen_cfg.early_stopping = True

    return tokenizer, model


def translate_to_english(text: str, tokenizer, model) -> str:
    """Translate text to English if needed."""
    try:
        lang = detect(text)
        if lang == "en":
            return text
    except Exception:
        pass
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            gen = model.generate(**inputs)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return out
    except Exception:
        return text  # fallback: return original


# ================================================================
# Step B: Toxic-BERT
# ================================================================

TOX_LABELS = ["tox_toxic", "tox_severe", "tox_obscene",
              "tox_threat", "tox_insult", "tox_identity"]


def load_toxic_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def get_toxicity_scores(text: str, tokenizer, model) -> Dict[str, float]:
    if not isinstance(text, str) or text.strip() == "":
        return {label: 0.0 for label in TOX_LABELS}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.sigmoid(out.logits).cpu().numpy()[0]

    return {name: float(p) for name, p in zip(TOX_LABELS, probs)}


def compute_pred_tox_final(tox_dict: Dict[str, float]) -> int:
    """Apply your thresholds to get final toxicity."""
    toxic_score = tox_dict["tox_toxic"]
    other_labels = ["tox_severe", "tox_obscene",
                    "tox_threat", "tox_insult", "tox_identity"]
    if toxic_score > 0.85:
        return 1
    if any(tox_dict[f] > 0.3 for f in other_labels):
        return 1
    return 0


# ================================================================
# Step C: Adult classifier
# ================================================================

def load_adult_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_adult_content(text: str, tokenizer, model, device):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, 0

    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True,
                       max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()[0][0]

    prob = float(1 / (1 + np.exp(-logits)))
    label = int(prob > 0.5)
    return prob, label


# ================================================================
# Step D: DeBERTa attack classifier (only for toxic/sexual)
# ================================================================

def load_attack_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_attack(context: str, tokenizer, model, device):
    if not isinstance(context, str) or context.strip() == "":
        return 0.0, 0

    inputs = tokenizer(context, return_tensors="pt",
                       truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.cpu().numpy()[0]

    prob = 1 / (1 + np.exp(-logits))[0]
    label = int(prob > 0.5)
    return float(prob), label


# ================================================================
# Step E: Final 3-level label
# ================================================================

def compute_final_label(pred_tox_adult_final: int, pred_attack_label: int) -> str:
    if pred_tox_adult_final == 0:
        return "Neutral"
    else:
        if pred_attack_label == 0:
            return "Profanity"
        else:
            return "Targeted"


# ================================================================
# Main
# ================================================================

def process_prompt_csv(
    prompt_csv: str,
    output_csv: str,
    trans_model_dir: str,
    toxic_model_dir: str,
    adult_model_name: str,
    attack_model_dir: str,
):
    """
    Input: prompt.csv (url, text, context)
    Output: label_detail.csv
    """

    df = pd.read_csv(prompt_csv)

    # ======================
    # Load all models once
    # ======================
    print("Loading translation model...")
    trans_tokenizer, trans_model = load_translation_model(trans_model_dir)
    trans_model.generation_config.num_beams = 5
    trans_model.generation_config.max_length = 256
    trans_model.generation_config.early_stopping = True

    print("Loading toxic-bert model...")
    tox_tokenizer, tox_model = load_toxic_model(toxic_model_dir)

    print("Loading adult classifier...")
    adult_tokenizer, adult_model, adult_device = load_adult_model(adult_model_name)

    print("Loading DeBERTa attack classifier...")
    atk_tokenizer, atk_model, atk_device = load_attack_model(attack_model_dir)

    # =====================================================
    # Process each row
    # =====================================================
    text_ens = []
    pred_tox = []
    pred_sex_prob = []
    pred_sex_label = []
    pred_attack_prob_list = []
    pred_attack_label_list = []
    pred_tox_adult_final_list = []
    final_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        text = norm(row["text"])
        context = norm(row["context"])

        # ---- Step A: translate ----
        text_en = translate_to_english(text, trans_tokenizer, trans_model)
        text_ens.append(text_en)

        # ---- Step B: toxicity scores ----
        tox_dict = get_toxicity_scores(text_en, tox_tokenizer, tox_model)
        pred_tox.append(tox_dict)

        # ---- Step C: adult ----
        a_prob, a_label = predict_adult_content(
            text_en, adult_tokenizer, adult_model, adult_device)
        pred_sex_prob.append(a_prob)
        pred_sex_label.append(a_label)

        # ---- Step D: combine toxicity + adult ----
        tox_final = compute_pred_tox_final(tox_dict)
        pred_tox_adult_final = int(tox_final or a_label)
        pred_tox_adult_final_list.append(pred_tox_adult_final)

        # ---- Step E: attack classifier (only for toxic/sexual) ----
        if pred_tox_adult_final == 1:
            atk_prob, atk_label = predict_attack(
                context, atk_tokenizer, atk_model, atk_device)
        else:
            atk_prob, atk_label = 0.0, 0

        pred_attack_prob_list.append(atk_prob)
        pred_attack_label_list.append(atk_label)

        # ---- Step F: final 3-level label ----
        final_label_level = compute_final_label(
            pred_tox_adult_final, atk_label)
        final_labels.append(final_label_level)

    # =====================================================
    # Organize output dataframe
    # =====================================================

    # Toxicity scores split
    tox_cols = {k: [d[k] for d in pred_tox] for k in TOX_LABELS}

    out_df = pd.DataFrame({
        "url": df["url"].astype(str),
        "text": df["text"].astype(str),
        "context": df["context"].astype(str),
        "text_en": text_ens,

        "pred_tox_toxic": tox_cols["tox_toxic"],
        "pred_tox_severe": tox_cols["tox_severe"],
        "pred_tox_obscene": tox_cols["tox_obscene"],
        "pred_tox_threat": tox_cols["tox_threat"],
        "pred_tox_insult": tox_cols["tox_insult"],
        "pred_tox_identity": tox_cols["tox_identity"],
    })

    # Add toxicity final
    out_df["pred_tox_final"] = [
        compute_pred_tox_final(d) for d in pred_tox
    ]

    # Add adult
    out_df["pred_adult_prob"] = pred_sex_prob
    out_df["pred_adult_label"] = pred_sex_label

    # combined
    out_df["pred_tox_adult_final"] = pred_tox_adult_final_list

    # Attack classifier
    out_df["pred_attack_prob"] = pred_attack_prob_list
    out_df["pred_attack_label"] = pred_attack_label_list

    # Final 3-level label
    out_df["final_label_level"] = final_labels

    # Save
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] label_detail.csv saved to {output_csv}")


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    PROMPT_CSV = "prompt.csv"
    OUTPUT_CSV = "label_detail.csv"

    # Paths to your models
    TRANS_MODEL_DIR = "../../small100"  # small100_tokenizer & small100_model
    TOXIC_MODEL_DIR = "../../toxic-bert-model"
    ADULT_MODEL_NAME = "lazyghost/bert-large-uncased-Adult-Text-Classifier"
    ATTACK_MODEL_DIR = "../../attack_classifier_deberta_final"

    process_prompt_csv(
        prompt_csv=PROMPT_CSV,
        output_csv=OUTPUT_CSV,
        trans_model_dir=TRANS_MODEL_DIR,
        toxic_model_dir=TOXIC_MODEL_DIR,
        adult_model_name=ADULT_MODEL_NAME,
        attack_model_dir=ATTACK_MODEL_DIR,
    )
