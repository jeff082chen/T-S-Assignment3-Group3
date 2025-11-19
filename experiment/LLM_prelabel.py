import re
import json
import time
import os
import pandas as pd
from tqdm import tqdm
import ollama
from typing import Optional, Dict, List

# ---------- Configuration ----------

MODEL = "llama3.1:70b"
ENABLE_REDACTION = False  # 不建議啟用,會影響準確性
MAX_RETRIES = 2
RETRY_DELAY = 2.0
SAVE_EVERY = 50  # 每處理多少筆保存一次

# ---------- Utilities ----------

def normalize_str(v):
    """Safely convert any cell value to clean string."""
    if isinstance(v, float) or pd.isna(v):
        return ""
    return str(v).strip()

def extract_json_block(text: str) -> Optional[str]:
    """Extract JSON-like block from model output."""
    if not text:
        return None
    
    # Strategy 1: Markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Strategy 2: JSON with expected keys
    match = re.search(
        r"(\{[^{}]*(?:is_attack|attack_prob)[^{}]*\})",
        text,
        re.DOTALL
    )
    if match:
        return match.group(1)
    
    # Strategy 3: Plain JSON
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    
    return None

def check_ollama_available() -> bool:
    """Check if Ollama service is running."""
    try:
        ollama.list()
        return True
    except Exception as e:
        print(f"❌ Ollama service not available: {e}")
        print("💡 Please start Ollama: 'ollama serve'")
        return False

# ---------- Prompt construction ----------

def make_targeted_attack_prompt(row: pd.Series) -> str:
    """Build prompt for targeted attack detection."""
    author = normalize_str(row.get("author_handle", ""))
    text = normalize_str(row.get("text", ""))
    quoted_author = normalize_str(row.get("quoted_author_handle", ""))
    quoted_text = normalize_str(row.get("quoted_text", ""))
    mentions = normalize_str(row.get("mentions_handle", ""))
    reply_author = normalize_str(row.get("parent_author_handle", ""))
    reply_text = normalize_str(row.get("parent_text", ""))
    tags = normalize_str(row.get("tags", ""))

    context_parts = []

    context_parts.append(f"Author: @{author}")
    context_parts.append(f"Post: {text}")

    if reply_author or reply_text:
        reply_info = "Reply to "
        if reply_author:
            reply_info += f"@{reply_author}: "
        if reply_text:
            reply_info += reply_text
        context_parts.append(reply_info)

    if quoted_author or quoted_text:
        quoted_info = "Quoted post"
        if quoted_author:
            quoted_info += f" by @{quoted_author}"
        if quoted_text:
            quoted_info += f": {quoted_text}"
        context_parts.append(quoted_info)

    if mentions:
        context_parts.append(f"Mentioned users (@handle): {mentions}")

    if tags:
        context_parts.append(f"Tags: {tags}")

    context = "\n".join(context_parts) if context_parts else "Post: [empty]"

    prompt = f"""You are an AI content moderation model for research purposes.

Analyze the social media post and classify targeted attacks. Do not repeat offensive content.

The term "profanity" refers strictly to: Explicit swear words, racial/ethnic slurs, sexually explicit insults, or violent threats.
The term "attack" is defined as the usage of such profanity specifically to **insult, degrade, mock, or threaten** a target. The context must be hostile or aggressive.

**CRITICAL EXCLUSION (Positive/Colloquial Use):**
Usage of profanity is **NOT** an attack if:
1. It acts as an **intensifier for positive sentiment** (e.g., "This is fucking amazing", "Damn, good job").
2. It is used as a **term of endearment, slang, or friendly banter** within a clearly positive or admiring context (e.g., "She is a bad bitch", "Love you crazy mofos").

Determine if the post contains attacks in these categories:
1.⁠ ⁠*attack_by_quote* — The post quoted another post, and insults/mocks/threatens "the author" of the quoted post, with profanity words.
2.⁠ ⁠*attack_by_mention* — The post mention some other account with "@", and insults/mocks/threatens that account, with profanity words.
3.⁠ ⁠*attack_by_reply* —The post reply another post, and insults/mocks/threatens "the author" of the replied post, with profanity words.
4.⁠ ⁠*attack_named_person* — The post mentions a specific person or a clearly identifiable title referring to someone, and insults/mocks/threatens that individual, with profanity words.
5.⁠ ⁠*attack_proper_noun* — The post contains insults, mockery, or aggressive/profane language directed at a proper noun that is not an individual person, such as an organization, company, political party, nationality, religion, event, location, or other named entity.

If any of above appears to be true, then is an attack.

Return ONLY a valid JSON object with lowercase booleans. No explanation.

Required format:
{{
  "is_attack": "true" / "false",
  "attack_prob" float, # logits between 0–1
}}

Context:
{context}
"""
    return prompt.strip()

# ---------- Core LLM call ----------

def analyze_post(prompt: str, retries: int = MAX_RETRIES, delay: float = RETRY_DELAY) -> Dict:
    """Call Ollama locally and parse JSON output with retry logic."""
    default_result = {
        "is_attack": "false",
        "attack_prob": 0.0,
    }
    
    for attempt in range(retries + 1):
        try:
            res = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a content moderation classifier. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 150,
                }
            )
            
            text_out = res["message"]["content"].strip()
            
            json_str = extract_json_block(text_out)
            if not json_str:
                raise ValueError(f"No JSON found in output: {text_out[:80]}...")
            
            result = json.loads(json_str)

            # Validate and set defaults
            parsed_result = {}

            is_attack_val = result.get("is_attack", default_result["is_attack"])
            if isinstance(is_attack_val, bool):
                parsed_result["is_attack"] = str(is_attack_val).lower()
            else:
                parsed_result["is_attack"] = str(is_attack_val).strip().lower()
            if parsed_result["is_attack"] not in {"true", "false"}:
                parsed_result["is_attack"] = default_result["is_attack"]

            attack_prob_val = result.get("attack_prob", default_result["attack_prob"])
            try:
                attack_prob_float = float(attack_prob_val)
            except (TypeError, ValueError):
                attack_prob_float = default_result["attack_prob"]
            parsed_result["attack_prob"] = max(0.0, min(1.0, attack_prob_float))

            return parsed_result

        except Exception as e:
            if attempt < retries:
                print(f"⚠️ Error (attempt {attempt + 1}/{retries + 1}): {e}")
                time.sleep(delay)
                continue
            else:
                print(f"❌ Final error: {e}")
                return default_result
    
    return default_result

# ---------- Checkpoint management ----------

def get_checkpoint_file(output_csv: str) -> str:
    """Get checkpoint filename for resume capability."""
    base, ext = os.path.splitext(output_csv)
    return f"{base}_checkpoint{ext}"

def load_checkpoint(checkpoint_file: str) -> tuple[pd.DataFrame, int]:
    """
    Load checkpoint to resume interrupted processing.
    Returns: (partial_dataframe, last_processed_index)
    """
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file, encoding='utf-8')
            required_cols = ["url", "is_attack", "attack_prob"]
            if all(col in df_checkpoint.columns for col in required_cols):
                complete_rows = df_checkpoint[
                    df_checkpoint["url"].notna() &
                    df_checkpoint["is_attack"].notna() &
                    df_checkpoint["attack_prob"].notna()
                ]
                if len(complete_rows) > 0:
                    last_idx = len(complete_rows)
                    print(f"📂 Checkpoint found: {last_idx} rows already processed")
                    return df_checkpoint.iloc[:last_idx], last_idx
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
    
    return None, 0

def save_partial(output_csv: str, df: pd.DataFrame, results: List[Dict], start_idx: int = 0):
    """
    Save partial results with checkpoint capability.
    
    Args:
        output_csv: Output file path
        df: Original dataframe
        results: List of classification results
        start_idx: Starting index (for resume support)
    """
    if not results:
        return
    
    processed_count = len(results)
    total_processed = start_idx + processed_count
    
    # Create results dataframe
    res_df = pd.DataFrame(results)
    if "url" not in df.columns:
        raise KeyError("Input dataframe must contain a 'url' column.")

    url_slice = df.iloc[start_idx:total_processed][["url"]].reset_index(drop=True)
    out_df = pd.concat(
        [
            url_slice,
            res_df.reset_index(drop=True)
        ],
        axis=1
    )
    
    # If resuming, combine with previous checkpoint
    checkpoint_file = get_checkpoint_file(output_csv)
    if start_idx > 0 and os.path.exists(checkpoint_file):
        try:
            prev_df = pd.read_csv(checkpoint_file, encoding='utf-8')
            out_df = pd.concat([prev_df, out_df], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not merge with previous checkpoint: {e}")
    
    # Save to checkpoint file
    out_df.to_csv(checkpoint_file, index=False, encoding='utf-8-sig')
    print(f"💾 Checkpoint saved: {total_processed}/{len(df)} rows")

def finalize_output(output_csv: str):
    """Move checkpoint file to final output file."""
    checkpoint_file = get_checkpoint_file(output_csv)
    if os.path.exists(checkpoint_file):
        try:
            # Read checkpoint
            df = pd.read_csv(checkpoint_file, encoding='utf-8')
            # Save as final output
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            # Remove checkpoint
            os.remove(checkpoint_file)
            print(f"✅ Final output saved to {output_csv}")
            return True
        except Exception as e:
            print(f"❌ Failed to finalize output: {e}")
            return False
    return False

# ---------- Main analysis function ----------

def analyse_and_save(input_csv: str, output_csv: str, save_every: int = SAVE_EVERY, resume: bool = True):
    """
    Main analysis function with checkpoint and resume support.
    
    Args:
        input_csv: Input CSV file path
        output_csv: Output CSV file path
        save_every: Save checkpoint every N rows
        resume: Whether to resume from checkpoint if exists
    """
    
    # Pre-flight checks
    print("🔍 Checking Ollama availability...")
    if not check_ollama_available():
        return
    
    # Load input data
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
        print(f"✅ Loaded {len(df)} rows from {input_csv}")
    except Exception as e:
        print(f"❌ Failed to read input CSV: {e}")
        return
    
    # Check for checkpoint
    start_idx = 0
    existing_results = []
    
    if resume:
        checkpoint_df, start_idx = load_checkpoint(get_checkpoint_file(output_csv))
        if start_idx > 0:
            user_input = input(f"Resume from row {start_idx}? (y/n): ").strip().lower()
            if user_input != 'y':
                start_idx = 0
                print("Starting from beginning...")
            else:
                print(f"Resuming from row {start_idx}")
    
    # Process posts
    results = []
    total_rows = len(df)
    
    print(f"\n🚀 Starting analysis with model: {MODEL}")
    print(f"⚙️ Redaction: {'Enabled' if ENABLE_REDACTION else 'Disabled'}")
    print(f"📊 Processing: {start_idx}/{total_rows} → {total_rows}")
    
    try:
        for idx in tqdm(range(start_idx, total_rows), desc="Analyzing posts", initial=start_idx, total=total_rows):
            row = df.iloc[idx]
            prompt = make_targeted_attack_prompt(row)
            res = analyze_post(prompt)
            results.append(res)
            
            # Save checkpoint periodically
            if len(results) % save_every == 0:
                save_partial(output_csv, df, results, start_idx)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user. Saving progress...")
        save_partial(output_csv, df, results, start_idx)
        print(f"💾 Progress saved. Run again to resume from row {start_idx + len(results)}")
        return
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        save_partial(output_csv, df, results, start_idx)
        print(f"💾 Progress saved despite error")
        return
    
    # Final save
    save_partial(output_csv, df, results, start_idx)
    finalize_output(output_csv)
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"📊 Analysis Complete - Statistics")
    print(f"{'='*60}")
    print(f"  Total posts processed: {len(results)}")
    
    if results:
        res_df = pd.DataFrame(results)
        attack_mask = res_df["is_attack"].astype(str).str.lower() == "true"
        attack_count = attack_mask.sum()
        attack_percentage = (attack_count / len(res_df) * 100) if len(res_df) > 0 else 0
        avg_prob = res_df["attack_prob"].mean() if "attack_prob" in res_df.columns else 0.0

        print(f"\n📈 Attack Detection Results:")
        print(f"  is_attack=true rows: {attack_count} ({attack_percentage:.2f}%)")
        print(f"  Average attack_prob: {avg_prob:.3f}")

# ---------- Entry point ----------

if __name__ == "__main__":
    input_csv = "final_version/train_features.csv"
    output_csv = "final_version/train_llama_result.csv"
    
    print("=" * 60)
    print("  Targeted Attack Detection using Ollama")
    print("=" * 60)
    
    analyse_and_save(
        input_csv=input_csv,
        output_csv=output_csv,
        save_every=SAVE_EVERY,
        resume=True  # 啟用斷點續傳
    )
    
    print("\n" + "=" * 60)
    print("  Processing Complete")
    print("=" * 60)
