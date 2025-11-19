import pandas as pd
from typing import Any

# =========================
# Utility
# =========================

def norm(v: Any) -> str:
    """Normalize any cell to a clean string (no NaN)."""
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


# =========================
# Build context from features
# =========================

def build_context(row: pd.Series) -> str:
    """Construct the context string for one Bluesky post."""

    # Extract fields safely
    author = norm(row.get("author_handle"))
    text = norm(row.get("text"))
    quoted_author = norm(row.get("quoted_author_handle"))
    quoted_text = norm(row.get("quoted_text"))
    mentions = norm(row.get("mentions_handle"))
    parent_author = norm(row.get("parent_author_handle"))
    parent_text = norm(row.get("parent_text"))
    tags = norm(row.get("tags"))
    status = norm(row.get("status"))

    # If Step 1 failed → produce a minimal context
    if status != "success":
        return "[ERROR_POST]"

    parts = []

    # Author + Post
    if author:
        parts.append(f"Author: @{author}")
    else:
        parts.append("Author: [unknown]")

    parts.append(f"Post: {text if text else '[empty]'}")

    # Reply context
    if parent_author or parent_text:
        s = "Reply to "
        if parent_author:
            s += f"@{parent_author}"
        if parent_text:
            s += f": {parent_text}"
        parts.append(s)

    # Quoted context
    if quoted_author or quoted_text:
        s = "Quoted"
        if quoted_author:
            s += f" @{quoted_author}"
        if quoted_text:
            s += f": {quoted_text}"
        parts.append(s)

    # Mentions
    if mentions:
        parts.append(f"Mentions: {mentions}")

    # Tags
    if tags:
        parts.append(f"Tags: {tags}")

    return "\n".join(parts)


# =========================
# Main: features.csv → prompt.csv
# =========================

def make_prompts(
    features_csv: str,
    output_csv: str
):
    """
    Load features.csv from Step 1, convert to context format,
    and produce prompt.csv with schema:
      url, text, context
    """
    df = pd.read_csv(features_csv)

    required_cols = [
        "url", "text", "status",
        "author_handle", "quoted_author_handle", "quoted_text",
        "mentions_handle", "parent_author_handle", "parent_text", "tags"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in features.csv: {col}")

    contexts = []
    for _, row in df.iterrows():
        ctx = build_context(row)
        contexts.append(ctx)

    out_df = pd.DataFrame({
        "url": df["url"].astype(str),
        "text": df["text"].astype(str),
        "context": contexts
    })

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] prompt.csv saved to {output_csv}")


# =========================
# Entry
# =========================

if __name__ == "__main__":
    FEATURES_CSV = "features.csv"          # Step 1 output
    OUTPUT_CSV = "prompt.csv"              # Step 2 output

    make_prompts(FEATURES_CSV, OUTPUT_CSV)
