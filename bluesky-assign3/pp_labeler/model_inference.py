"""
model_inference.py
------------------

Unified ML inference module for Assignment 3.

Implements the full logic used in your step3_label_detail.py pipeline:

    1. Translation → English (NLLB)
    2. Toxicity scoring (unitary/toxic-bert, 6 dimensions)
    3. Adult sexual/profanity classification (bert-large-uncased-Adult-Text-Classifier)
    4. Targeted attack classification (distilled DeBERTa-v3-small)

This file exposes four public functions:

    translate_to_english(text)
    compute_toxicity_scores(text_en)
    compute_adult_toxicity_flag(text_en, tox_scores)
    compute_attack_label(context)
"""

import os
import torch
from langdetect import detect
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ======================================================
#  GLOBAL CONFIG (EXTRACTED THRESHOLDS)
# ======================================================

# Toxic-BERT thresholds (from original Step3 logic)
TOX_TOXIC_THRESHOLD = 0.45
TOX_OTHER_THRESHOLD = 0.12

# Adult classifier threshold (original Step3: prob > 0.5)
ADULT_THRESHOLD = 0.50

# Attack classifier threshold (original Step3: 0.5)
ATTACK_THRESHOLD = 0.75

# ======================================================
# Paths
# ======================================================
if not os.path.exists("models"):
    os.makedirs("models")
LOCAL_NLLB_DIR   = "models/nllb/"
LOCAL_TOXIC_DIR  = "models/toxic-bert/"
LOCAL_ADULT_DIR  = "models/adult-bert/"
LOCAL_ATTACK_DIR = "models/attack-classifier/"

# torch device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Utility function: load local or download
# ======================================================
def load_local_or_remote(local_dir, remote_id, model_class):
    """
    Try loading local folder.
    If missing → download from HuggingFace → save to local dir → return model.
    """
    try:
        if os.path.isdir(local_dir):
            print(f"[INFO] Loading local model at {local_dir}")
            tok = AutoTokenizer.from_pretrained(local_dir)
            model = model_class.from_pretrained(local_dir)
            return tok, model

        # ---- fallback to HF download ----
        print(f"[INFO] Local folder missing: {local_dir}")
        print(f"[INFO] Downloading model from HF: {remote_id} ...")

        tok = AutoTokenizer.from_pretrained(remote_id)
        model = model_class.from_pretrained(remote_id)

        os.makedirs(local_dir, exist_ok=True)
        tok.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

        print(f"[INFO] Saved downloaded model to {local_dir}")
        return tok, model

    except Exception as e:
        print(f"[ERROR] Failed to load model ({local_dir} / {remote_id}): {e}")
        return None, None


# ======================================================
# 1. NLLB translator
# ======================================================
nllb_tokenizer, nllb_model = load_local_or_remote(
    LOCAL_NLLB_DIR,
    "alirezamsh/small100",
    AutoModelForSeq2SeqLM
)


# ======================================================
# 2. Toxic-BERT
# ======================================================
toxic_tokenizer, toxic_model = load_local_or_remote(
    LOCAL_TOXIC_DIR,
    "unitary/toxic-bert",
    AutoModelForSequenceClassification
)


# ======================================================
# 3. Adult content classifier
# ======================================================
adult_tokenizer, adult_model = load_local_or_remote(
    LOCAL_ADULT_DIR,
    "lazyghost/bert-large-uncased-Adult-Text-Classifier",
    AutoModelForSequenceClassification
)


# ======================================================
# 4. Distilled attack classifier (your model)
# ======================================================
print("[INFO] Loading attack classifier...")

attack_tokenizer, attack_model = load_local_or_remote(
    LOCAL_ATTACK_DIR,
    "jeff082chen/attack-classifier-deberta-context",   # fallback to your HF repo
    AutoModelForSequenceClassification
)

if attack_model is None:
    raise RuntimeError(
        f"[FATAL] Cannot load attack classifier.\n"
        f"Tried: local={LOCAL_ATTACK_DIR}, remote=jeff082chen/attack-classifier-deberta-context\n"
        f"→ Ensure that your model exists on HF or include it locally."
    )

print("[INFO] Attack classifier ready.")

# -------------------------
# Move models to device
# -------------------------
if nllb_model:
    nllb_model.to(device)
if toxic_model:
    toxic_model.to(device)
if adult_model:
    adult_model.to(device)
if attack_model:
    attack_model.to(device)

# ======================================================
# 1. NLLB translator
# ======================================================

def translate_to_english(text: str) -> str:
    """
    Add lang detection + match original Step3 translation behavior
    """
    if not isinstance(text, str):
        return ""

    # ---- NEW: Detect language ----
    try:
        lang = detect(text)
        if lang == "en":
            return text
    except Exception:
        pass

    if not nllb_model:
        return text  # fallback

    inputs = nllb_tokenizer(text, return_tensors="pt").to(device)

    try:
        generated = nllb_model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=5,
            early_stopping=True,
            forced_bos_token_id=nllb_tokenizer.lang_code_to_id["eng_Latn"]
        )
        return nllb_tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception:
        return text


# ======================================================
# 2. TOXICITY SCORING
# ======================================================

TOXICITY_DIMENSIONS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "insult",
    "identity_attack",
    "threat"
]

def compute_toxicity_scores(text_en: str) -> dict:
    if not toxic_model:
        return {k: 0.0 for k in TOXICITY_DIMENSIONS}

    enc = toxic_tokenizer(
        text_en,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = toxic_model(**enc).logits.sigmoid().cpu().numpy()[0]

    return {dim: float(score) for dim, score in zip(TOXICITY_DIMENSIONS, logits)}


# ======================================================
# 3. ADULT CONTENT / PROFANITY (RESTORE ORIGINAL LOGIC)
# ======================================================

def compute_adult_toxicity_flag(text_en: str, tox: dict) -> bool:
    """
    EXACT matching Step3 logic:
       tox_final = compute_pred_tox_final
       pred_tox_adult_final = tox_final OR adult_label
    """

    # ---- Step3: compute_pred_tox_final ----
    tox_final = (
         tox["toxicity"] > TOX_TOXIC_THRESHOLD
         or any(tox[k] > TOX_OTHER_THRESHOLD for k in [
             "severe_toxicity",
             "obscene",
             "threat",
             "insult",
             "identity_attack",
         ])
    )

    # ---- Adult classifier → prob > 0.5 ----
    if not adult_model:
        adult_flag = False
    else:
        enc = adult_tokenizer(
            text_en,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            logits = adult_model(**enc).logits
            prob = torch.sigmoid(logits)[0].item() if logits.shape[-1] == 1 else torch.softmax(logits, dim=-1)[0,1].item()

        adult_flag = (prob > ADULT_THRESHOLD)

    # ---- Step3 final ----
    return bool(tox_final or adult_flag)


# ======================================================
# 4. ATTACK CLASSIFIER (RESTORE SINGLE-LOGIT BEHAVIOR)
# ======================================================

def compute_attack_label(context: str) -> bool:
    if not attack_model:
        return False

    enc = attack_tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = attack_model(**enc).logits

    # ORIGINAL Step3 uses 1-logit sigmoid → we enforce that
    if logits.shape[-1] == 1:
        prob = torch.sigmoid(logits)[0, 0].item()
    else:
        # enforce equivalent transformation: use only logit[1]
        prob = torch.sigmoid(logits[0, 1]).item()

    return prob > ATTACK_THRESHOLD
