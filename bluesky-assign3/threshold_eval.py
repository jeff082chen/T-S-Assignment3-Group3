import pandas as pd
from typing import Tuple, Dict

# ===========================================================
# Threshold configuration
# ===========================================================

TOX_TOXIC_THRESHOLD = 0.45
TOX_OTHER_THRESHOLD = 0.12
ADULT_THRESHOLD = 0.50
ATTACK_THRESHOLD = 0.75

# attack classifier
ENABLE_ATTACK_IF_MENTION = False
ENABLE_ATTACK_IF_QUOTE = False
ENABLE_ATTACK_IF_LOW_TOX = False
LOW_TOX_ATTACK_TRIGGER = 0.05


# ===========================================================
# Core classification logic
# ===========================================================

def classify_row(row) -> str:
    """
    依照 threshold 對單一 row 做分類。
    輸出: 'neutral', 'profanity', 'targeted'
    """

    # ---------- Step 1 ----------
    toxic = (
        row["tox_toxicity"] >= TOX_TOXIC_THRESHOLD
        or row["tox_severe_toxicity"] >= TOX_OTHER_THRESHOLD
        or row["tox_obscene"] >= TOX_OTHER_THRESHOLD
        or row["tox_insult"] >= TOX_OTHER_THRESHOLD
        or row["tox_identity_attack"] >= TOX_OTHER_THRESHOLD
        or row["tox_threat"] >= TOX_OTHER_THRESHOLD
        or row["adult_prob"] >= ADULT_THRESHOLD
    )

    if toxic:
        profanity_flag = True
    else:
        profanity_flag = False

    # ---------- Step 2 ----------
    should_run_attack = False

    if profanity_flag:
        should_run_attack = True

    if ENABLE_ATTACK_IF_MENTION and row["mention_count"] > 0:
        should_run_attack = True

    if ENABLE_ATTACK_IF_QUOTE and row["has_quote"]:
        should_run_attack = True

    if ENABLE_ATTACK_IF_LOW_TOX and row["tox_toxicity"] > LOW_TOX_ATTACK_TRIGGER:
        should_run_attack = True

    # ---------- Step 3: attack classification ----------
    targeted_flag = False
    if should_run_attack and row["attack_prob"] >= ATTACK_THRESHOLD:
        targeted_flag = True

    # ---------- Step 4: combine categories ----------
    if not profanity_flag and not targeted_flag:
        return "neutral"

    if profanity_flag and not targeted_flag:
        return "profanity"

    if targeted_flag:
        return "targeted"

    return "neutral"


# ===========================================================
# Metric functions
# ===========================================================

def precision_recall(pred, truth, positive_label) -> Tuple[float, float]:
    tp = sum((pred == positive_label) & (truth == positive_label))
    fp = sum((pred == positive_label) & (truth != positive_label))
    fn = sum((pred != positive_label) & (truth == positive_label))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return precision, recall


# ===========================================================
# Main evaluation
# ===========================================================

def evaluate_thresholds(csv_path: str):
    df = pd.read_csv(csv_path)

    df["pred_recalc"] = df.apply(classify_row, axis=1)

    # accuracy
    accuracy = (df["pred_recalc"] == df["gt_label"]).mean()

    # -------------------------
    # Neutral vs Toxic
    # -------------------------
    df["gt_toxic"] = df["gt_label"].isin(["profanity", "targeted"])
    df["pred_toxic"] = df["pred_recalc"].isin(["profanity", "targeted"])

    prec1, rec1 = precision_recall(
        df["pred_toxic"],
        df["gt_toxic"],
        True
    )

    # -------------------------
    # Profanity vs Targeted
    # -------------------------
    toxic_df = df[df["gt_toxic"]].copy()

    prec2, rec2 = precision_recall(
        toxic_df["pred_recalc"] == "targeted",
        toxic_df["gt_label"] == "targeted",
        True
    )

    print("===================================================")
    print("Threshold Evaluation Results")
    print("===================================================\n")

    print(f"Accuracy: {accuracy:.4f}\n")

    print("▶ Neutral vs Toxic")
    print(f"  Precision: {prec1:.4f}")
    print(f"  Recall:    {rec1:.4f}\n")

    print("▶ Profanity vs Targeted")
    print(f"  Precision: {prec2:.4f}")
    print(f"  Recall:    {rec2:.4f}\n")

    return df


# ===========================================================
# Run
# ===========================================================

if __name__ == "__main__":
    evaluate_thresholds("debug_metrics.csv")
