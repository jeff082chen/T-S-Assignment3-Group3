# Targeted Profanity & Attack Detection Pipeline

## Neutral → Profanity → Targeted Multi-stage Labeling System

This project implements a multi-stage labeling pipeline for classifying social media posts into three categories:

- Neutral
- Profanity
- Targeted Attack

The system integrates data collection from Bluesky, multilingual translation, toxicity detection, LLM-assisted contextual attack reasoning, and a distilled lightweight attack classifier.

---

## Pipeline Overview

The full workflow consists of five independent Python scripts, each with clean I/O interfaces to ensure modularity and reproducibility.

1. `step1_fetch_features.py`
2. `step2_build_prompt.py`
3. `step3_label_detail.py`
4. `step4_eval_results.py`
5. `step5_performance_matrix.py`

Below is a concise description of each component.

---

## 1️⃣ Fetch contextual features from Bluesky API

- **Script:** `step1_fetch_features.py`
- **Input:** `ground_truth.csv` (`url`, `label`)
- **Process:**
  - Convert Bluesky URLs to `at://` URIs.
  - Fetch each post and metadata: author handle, text, quoted author/text, reply-to author/text, mentions, tags.
  - Store only the fields required for classification.
- **Output:** `features.csv` with columns `url`, `author_handle`, `text`, `quoted_author_handle`, `quoted_text`, `mentions_handle`, `parent_author_handle`, `parent_text`, `tags`.

---

## 2️⃣ Build LLM-ready contextual prompts

- **Script:** `step2_build_prompt.py`
- **Input:** `features.csv`
- **Process:**
  - Combine relevant fields into a contextual description (original text, quoted context, mentions, reply context, tags).
  - Generate a clean textual context string suitable for LLM or ML classifier input.
- **Output:** `prompt.csv` with columns `url`, `text`, `context`.

---

## 3️⃣ Translate, detect profanity, and detect attacks

- **Script:** `step3_label_detail.py`
- **Input:** `prompt.csv`
- **Process:**
  1. **Translation:** Use `facebook/nllb-small100` to translate non-English posts to English (`text_en`).
  2. **Profanity detection:** Apply `unitary/toxic-bert` (six toxicity scores) and `bert-large-uncased-Adult-Text-Classifier`; combine into `pred_tox_final` and `pred_tox_adult_final`.
  3. **Targeted attack detection:** When `pred_tox_adult_final` is `true`, pass the context string into the distilled attack classifier (fine-tuned `microsoft/deberta-v3-small`) to obtain `pred_attack_label`.
- **Output:** `label_detail.csv` with columns `url`, `text_en`, `pred_tox_final`, `pred_tox_adult_final`, `pred_attack_label`, `context`.

---

## 4️⃣ Assign final labels

- **Script:** `step4_eval_results.py`
- **Input:** `label_detail.csv`, `ground_truth.csv`
- **Process:**
  - Apply the three-way rule:
    - `pred_tox_adult_final = false` → Neutral
    - `pred_attack_label = false` → Profanity
    - Otherwise → Targeted
  - Produce final results alongside ground-truth labels.
- **Output:** `final_label.csv` with columns `url`, `final_label`, `ground_truth_label`.

---

## 5️⃣ Compute performance metrics

- **Script:** `step5_performance_matrix.py`
- **Input:** `final_label.csv`
- **Process:**
  1. **Neutral vs Toxic (Profanity + Targeted):** accuracy, precision, recall, F1, confusion matrix.
  2. **Profanity vs Targeted (toxic subset):** accuracy, precision, recall, F1, confusion matrix.
- **Output:** `performance_matrix.csv` with `url`, `attack_result` (binary), `profanity_result` (binary), and metrics printed to the console.

---

## Environment

- Recommended Python version: 3.11
- Install dependencies: `pip install -r requirements.txt`

---

## System Highlights

- High-quality contextual attack detection
- LLM-to-small-model distillation for efficient deployment
- Multilingual post handling
- Modular, reproducible architecture
- End-to-end explainability from raw URL to final label
