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
import numpy as np
import subprocess
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

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
print(f"[INFO] Loading attack classifier...")

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


# ============================================================
#               1. TRANSLATION
# ============================================================

def translate_to_english(text: str) -> str:
    """
    If NLLB is available: translate text into English.
    Otherwise: identity function.
    """
    if not nllb_model:
        return text  # fallback

    inputs = nllb_tokenizer(text, return_tensors="pt").to(device)

    # NOTE: Exact translation pipeline depends on model type.
    # If using NLLB text-to-text model, replace this with generate().
    try:
        generated = nllb_model.generate(
            inputs.input_ids,
            forced_bos_token_id=nllb_tokenizer.lang_code_to_id["eng_Latn"]
        )
        translation = nllb_tokenizer.decode(generated[0], skip_special_tokens=True)
        return translation
    except:
        # fallback
        return text


# ============================================================
#               2. TOXICITY SCORING
# ============================================================

TOXICITY_DIMENSIONS = ["toxicity", "severe_toxicity", "obscene", "insult",
                       "identity_attack", "threat"]

def compute_toxicity_scores(text_en: str) -> dict:
    """
    Returns a dict of toxicity scores.

    If Toxic-BERT is not loaded, return zeros.
    """
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


# ============================================================
#               3. ADULT CONTENT / PROFANITY FUSION
# ============================================================

# recommended threshold from your Step3 logic:
SEXUAL_EXPLICIT_THRESHOLD = 0.40
OBSCENE_THRESHOLD = 0.35

def compute_adult_toxicity_flag(text_en: str, tox_scores: dict) -> bool:
    """
    Decide whether the post is 'adult / profanity'.

    Combines Toxic-BERT + Adult-BERT model.

    Returns:
        True  → classify as profanity
        False → classify as neutral
    """
    # If adult classifier unavailable, fallback to toxicity thresholds
    if not adult_model:
        return (
            tox_scores["obscene"] > OBSCENE_THRESHOLD or
            tox_scores["toxicity"] > 0.5 or
            tox_scores["severe_toxicity"] > 0.3
        )

    # -------- Adult classifier prediction --------
    enc = adult_tokenizer(
        text_en,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = adult_model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()  # probability of "adult"

    # Fusion rule (example from your step3):
    return (
        prob > 0.55 or
        tox_scores["obscene"] > OBSCENE_THRESHOLD or
        tox_scores["identity_attack"] > 0.4
    )


# ============================================================
#               4. TARGETED ATTACK DETECTOR
# ============================================================

ATTACK_THRESHOLD = 0.55   # tune based on your validation

def compute_attack_label(context: str) -> bool:
    """
    Returns True/False from your distilled DeBERTa attack classifier.
    """
    if not attack_model:
        return False  # fallback

    enc = attack_tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        logits = attack_model(**enc).logits

    # Case 1: model outputs 2 logits → normal classifier
    if logits.shape[-1] == 2:
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

    # Case 2: model outputs 1 logit → sigmoid binary classifier
    elif logits.shape[-1] == 1:
        prob = torch.sigmoid(logits)[0, 0].item()

    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    return prob >= ATTACK_THRESHOLD
