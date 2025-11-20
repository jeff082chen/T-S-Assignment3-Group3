# debug_thresholds.py
"""
Debug tool for threshold tuning.

Reads the same input CSV as test_labeler.py (URL, Labels),
and for each row outputs:
  - ground truth label
  - predicted label (using current thresholds)
  - toxicity scores
  - adult prob/flag
  - attack prob/flag
  - some context features (mentions, reply, quote)

Result is saved as debug_metrics.csv
"""

import os
import json

import pandas as pd
import torch
from atproto import Client
from dotenv import load_dotenv

from pp_labeler import (
    fetch_single_post_features,
    build_context_string,
    translate_to_english,
    compute_toxicity_scores,
    compute_adult_toxicity_flag,
    compute_attack_label,
)

# 直接從 model_inference 匯入 thresholds & 模型物件
from pp_labeler.model_inference import (
    TOX_TOXIC_THRESHOLD,
    TOX_OTHER_THRESHOLD,
    ADULT_THRESHOLD,
    ATTACK_THRESHOLD,
    adult_tokenizer,
    adult_model,
    attack_tokenizer,
    attack_model,
    device,
)

load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")


# ---------------------------------------------------------------------
# Helpers: probability-level access
# ---------------------------------------------------------------------
def get_adult_prob(text_en: str) -> float:
    """Return adult classifier probability of 'adult/sexual/profanity'."""
    if adult_model is None:
        return 0.0

    if not isinstance(text_en, str):
        text_en = str(text_en or "")

    enc = adult_tokenizer(
        text_en,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = adult_model(**enc).logits

    # binary 或 multi-class 都處理
    if logits.shape[-1] == 1:
        prob = torch.sigmoid(logits)[0].item()
    else:
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

    return float(prob)


def get_attack_prob(context: str) -> float:
    """Return attack classifier probability of 'targeted attack'."""
    if attack_model is None:
        return 0.0

    if not isinstance(context, str):
        context = str(context or "")

    enc = attack_tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        logits = attack_model(**enc).logits

    # 你在 model_inference 的邏輯：
    if logits.shape[-1] == 1:
        prob = torch.sigmoid(logits)[0, 0].item()
    else:
        prob = torch.sigmoid(logits[0, 1]).item()

    return float(prob)


def canonical_label(label_list):
    """
    Convert ["profanity"] / ["targeted"] / [] into a simple string label.
    """
    if not label_list:
        return "neutral"
    # Assignment spec 應該最多一個 label
    return str(label_list[0])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 讀 test-data/data.csv（或其他路徑）
    input_csv = "test-data/data.csv"
    output_csv = "debug_metrics.csv"

    df = pd.read_csv(input_csv)

    if "URL" not in df.columns or "Labels" not in df.columns:
        raise ValueError("Expected columns 'URL' and 'Labels' in input CSV.")

    # Bluesky login
    client = Client()
    client.login(USERNAME, PW)

    rows = []

    for _, row in df.iterrows():
        url = row["URL"]
        try:
            expected_labels = json.loads(row["Labels"])
        except Exception:
            expected_labels = []
        gt_label = canonical_label(expected_labels)

        # ---------------------------------------------------------
        # 1. Fetch features
        # ---------------------------------------------------------
        try:
            features = fetch_single_post_features(url, client)
        except Exception as exc:
            print(f"[ERROR] fetch_single_post_features failed for {url}: {exc}")
            rows.append({
                "url": url,
                "gt_label": gt_label,
                "error": f"fetch_error: {exc}",
            })
            continue

        # 基本 context signal
        text = features.get("text", "") or ""
        quoted_author = features.get("quoted_author", "") or ""
        quoted_text = features.get("quoted_text", "") or ""
        parent_author = features.get("parent_author", "") or ""
        parent_text = features.get("parent_text", "") or ""
        mentions = features.get("mentions") or []
        tags = features.get("tags") or []

        has_mentions = len(mentions) > 0
        has_quote = bool(quoted_author or quoted_text)
        is_reply = bool(parent_author or parent_text)

        # ---------------------------------------------------------
        # 2. Build context & translation
        # ---------------------------------------------------------
        context = build_context_string(features)
        text_en = translate_to_english(text)

        # ---------------------------------------------------------
        # 3. Toxicity scores
        # ---------------------------------------------------------
        tox_scores = compute_toxicity_scores(text_en)

        # 手動 compute tox_final，對應原 Step3 的 compute_pred_tox_final
        tox_final = (
            tox_scores["toxicity"] > TOX_TOXIC_THRESHOLD
            or any(
                tox_scores[k] > TOX_OTHER_THRESHOLD
                for k in [
                    "severe_toxicity",
                    "obscene",
                    "threat",
                    "insult",
                    "identity_attack",
                ]
            )
        )

        # ---------------------------------------------------------
        # 4. Adult prob + flag
        # ---------------------------------------------------------
        adult_prob = get_adult_prob(text_en)
        adult_flag = adult_prob > ADULT_THRESHOLD

        # is_adult_toxic = tox_final OR adult_flag（跟 compute_adult_toxicity_flag 一致）
        is_adult_toxic = bool(tox_final or adult_flag)

        # 也可以直接檢查與 compute_adult_toxicity_flag 是否一致
        is_adult_toxic_fn = compute_adult_toxicity_flag(text_en, tox_scores)

        # ---------------------------------------------------------
        # 5. Attack prob + flag（只有在 is_adult_toxic 時才跑，模擬現在 pipeline）
        # ---------------------------------------------------------
        attack_prob = 0.0
        attack_flag = False
        if is_adult_toxic:
            attack_prob = get_attack_prob(context)
            attack_flag = attack_prob > ATTACK_THRESHOLD

        # pipeline 最終 predicted label（跟 PolicyProposalLabeler 的邏輯一樣）
        if not is_adult_toxic:
            pred_label = "neutral"
        else:
            if attack_flag:
                pred_label = "targeted"
            else:
                pred_label = "profanity"

        is_correct = (pred_label == gt_label)

        # ---------------------------------------------------------
        # collect row
        # ---------------------------------------------------------
        debug_row = {
            "url": url,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "correct": int(is_correct),

            # toxicity dims
            "tox_toxicity": tox_scores["toxicity"],
            "tox_severe_toxicity": tox_scores["severe_toxicity"],
            "tox_obscene": tox_scores["obscene"],
            "tox_insult": tox_scores["insult"],
            "tox_identity_attack": tox_scores["identity_attack"],
            "tox_threat": tox_scores["threat"],
            "tox_final_flag": int(tox_final),

            # adult
            "adult_prob": adult_prob,
            "adult_flag": int(adult_flag),

            # combined toxicity/adult
            "is_adult_toxic": int(is_adult_toxic),
            "is_adult_toxic_fn": int(is_adult_toxic_fn),

            # attack
            "attack_prob": attack_prob,
            "attack_flag": int(attack_flag),

            # simple context signals
            "has_mentions": int(has_mentions),
            "mention_count": len(mentions),
            "has_quote": int(has_quote),
            "is_reply": int(is_reply),
        }

        rows.append(debug_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved debug metrics to {output_csv}")


if __name__ == "__main__":
    main()