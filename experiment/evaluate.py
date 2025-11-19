import pandas as pd

# ================================================================
# Helpers
# ================================================================

def get_case_name(gt: int, pred: int) -> str:
    """Return tp/tn/fp/fn based on binary ground truth and prediction."""
    if gt == 1 and pred == 1:
        return "tp"
    elif gt == 0 and pred == 0:
        return "tn"
    elif gt == 0 and pred == 1:
        return "fp"
    elif gt == 1 and pred == 0:
        return "fn"
    else:
        return "unknown"


# ================================================================
# Main Evaluation
# ================================================================

def evaluate_predictions(label_detail_csv: str,
                        ground_truth_csv: str,
                        output_csv: str = "performance_matrix.csv"):
    """
    Using label_detail.csv and ground_truth.csv,
    compute evaluation results for:
        1. Neutral vs (Profanity + Targeted)
        2. Profanity vs Targeted (toxic rows only)
    """

    print("Loading files...")
    df_pred = pd.read_csv(label_detail_csv)
    df_gt = pd.read_csv(ground_truth_csv)

    # --- merge ---
    df = df_pred.merge(df_gt, on="url", how="inner", suffixes=("_pred", "_gt"))

    # Rename for clarity
    df = df.rename(columns={"label": "ground_truth_label",
                            "final_label_level": "predicted_label"})

    # ================================================================
    # 1️⃣ Binary classification: Neutral vs Toxic (Profanity + Targeted)
    # ================================================================
    # GT
    df["gt_profanity_flag"] = df["ground_truth_label"].map(
        lambda x: 0 if x == "Neutral" else 1
    )

    # Pred
    df["pred_profanity_flag"] = df["predicted_label"].map(
        lambda x: 0 if x == "Neutral" else 1
    )

    df["profanity_result"] = df.apply(
        lambda r: get_case_name(r["gt_profanity_flag"], r["pred_profanity_flag"]),
        axis=1
    )

    # ================================================================
    # 2️⃣ Binary classification: Profanity vs Targeted (toxic rows only)
    # ================================================================
    def map_attack_flag(x: str) -> int:
        if x == "Profanity":
            return 0
        elif x == "Targeted":
            return 1
        else:
            return -1  # Neutral → skip

    df["gt_attack_flag"] = df["ground_truth_label"].map(map_attack_flag)
    df["pred_attack_flag"] = df["predicted_label"].map(map_attack_flag)

    attack_results = []
    for _, r in df.iterrows():
        if r["gt_attack_flag"] == -1 or r["pred_attack_flag"] == -1:
            attack_results.append("skip")
        else:
            attack_results.append(
                get_case_name(r["gt_attack_flag"], r["pred_attack_flag"])
            )

    df["attack_result"] = attack_results

    # ================================================================
    # Save performance_matrix.csv
    # ================================================================
    out_cols = [
        "url",
        "ground_truth_label",
        "predicted_label",
        "profanity_result",
        "attack_result",
    ]

    df[out_cols].to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved performance_matrix.csv to {output_csv}")

    # ================================================================
    # Print statistics
    # ================================================================
    print("\n==============================")
    print("📊 Neutral vs Toxic (Profanity + Targeted)")
    print("==============================")

    prof_subset = df[df["profanity_result"] != "unknown"]
    if len(prof_subset) > 0:
        tn = (prof_subset["profanity_result"] == "tn").sum()
        tp = (prof_subset["profanity_result"] == "tp").sum()
        fp = (prof_subset["profanity_result"] == "fp").sum()
        fn = (prof_subset["profanity_result"] == "fn").sum()

        acc = (tp + tn) / len(prof_subset)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall /
              (precision + recall)) if (precision + recall) > 0 else 0

        print(f"Total samples: {len(prof_subset)}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

    print("\n==============================")
    print("📊 Profanity vs Targeted (Toxic-only rows)")
    print("==============================")

    atk_subset = df[df["attack_result"] != "skip"]
    if len(atk_subset) > 0:
        tn = (atk_subset["attack_result"] == "tn").sum()
        tp = (atk_subset["attack_result"] == "tp").sum()
        fp = (atk_subset["attack_result"] == "fp").sum()
        fn = (atk_subset["attack_result"] == "fn").sum()

        acc = (tp + tn) / len(atk_subset)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall /
              (precision + recall)) if (precision + recall) > 0 else 0

        print(f"Total toxic samples: {len(atk_subset)}")
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")


# ================================================================
# Run
# ================================================================
if __name__ == "__main__":
    evaluate_predictions(
        label_detail_csv="label_detail.csv",
        ground_truth_csv="ground_truth.csv",
        output_csv="performance_matrix.csv"
    )
