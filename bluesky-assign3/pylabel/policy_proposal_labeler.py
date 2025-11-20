"""
policy_proposal_labeler.py
===========================

This file implements the final labeler required for CS 5342 Assignment 3.

The main class `PolicyProposalLabeler` exposes a single method:

    moderate_post(url: str) -> List[str]

Given a Bluesky post URL, this method fetches the post, constructs
contextual features, performs multilingual toxicity + adult-content +
LLM-distilled targeted attack classification, and returns final labels
as required by the assignment.

All heavy logic is delegated to modules in the custom pipeline.
These modules should be implemented separately inside your project:

    - fetch_features.py
    - build_context.py
    - labeler.py  (toxicity model + attack model inference)
    - evaluate.py (optional helper utilities)

Only orchestration happens here.
"""

from typing import List, Optional

from atproto import Client

# --- Custom modules provided elsewhere in the project ---
from pp_labeler import (
    fetch_single_post_features,
    build_context_string,
    translate_to_english,
    compute_toxicity_scores,
    compute_adult_toxicity_flag,
    compute_attack_label,
)
# -------------------------------------------------------------------------


class PolicyProposalLabeler:
    """
    Final automated labeler class implementing targeted profanity detection.

    This class orchestrates the multi-stage pipeline:

        1. Fetch contextual post features via API
        2. Build clean context string (mentions, quotes, reply chain, etc.)
        3. Machine translation → English (NLLB)
        4. Toxicity detection via Toxic-BERT
        5. Adult/profanity detection via Adult-BERT classifier
        6. Targeted attack reasoning via distilled DeBERTa classifier
        7. Rule-based merging into final {neutral, profanity, targeted}

    Returns:
        ["targeted"]    → targeted harassment
        ["profanity"]   → profanity but not targeted
        []              → neutral
    """

    def __init__(self, client: Optional[Client] = None):
        """
        Initialize all necessary models and utilities.
        These are loaded inside the imported modules.
        """
        print("[Init] Loading models and pipeline components ...")
        self.client = client

    # =====================================================================
    # Main API required by the assignment
    # =====================================================================
    def moderate_post(self, url: str, client: Optional[Client] = None) -> List[str]:
        """
        Apply automated moderation to a Bluesky post at the given URL.

        Args:
            url (str): Full Bluesky post URL
            client (Client | None): Optional override Bluesky client.

        Returns:
            List[str]: A list containing one label ("profanity", "targeted")
                       or [] for neutral.
        """
        active_client = client or self.client
        if active_client is None:
            print("[Error] PolicyProposalLabeler requires a logged-in Client.")
            return []

        try:
            # ------------------------------------------------------------
            # 1. Fetch post features
            # ------------------------------------------------------------
            features = fetch_single_post_features(url, active_client)

            # ------------------------------------------------------------
            # 2. Build full contextual description
            # ------------------------------------------------------------
            context = build_context_string(features) or "Original post: "

            # ------------------------------------------------------------
            # 3. Translate to English (if needed)
            # ------------------------------------------------------------
            raw_text = features.get("text", "") if isinstance(features, dict) else ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text or "")
            text_en = translate_to_english(raw_text or "")

            # ------------------------------------------------------------
            # 4. Compute toxicity scores (Toxic-BERT)
            # ------------------------------------------------------------
            tox_scores = compute_toxicity_scores(text_en)

            # ------------------------------------------------------------
            # 5. Adult-content / profanity classification
            # ------------------------------------------------------------
            is_adult_toxic = compute_adult_toxicity_flag(text_en, tox_scores)

            # ------------------------------------------------------------
            # 6. If profanity → run targeted attack classifier
            # ------------------------------------------------------------
            pred_attack = False
            if is_adult_toxic:
                pred_attack = compute_attack_label(context)

            # ------------------------------------------------------------
            # 7. Rule-based merging into final label
            # ------------------------------------------------------------
            if not is_adult_toxic:
                return []                       # neutral

            if pred_attack:
                return ["targeted"]

            return ["profanity"]
        except Exception as exc:
            print(f"[Error] Moderation failed for {url}: {exc}")
            return []

    # =====================================================================
    # Optional helper for batch processing (not required by assignment)
    # =====================================================================
    def moderate_batch(self, url_list: List[str], client: Optional[Client] = None):
        results = {}
        for url in url_list:
            results[url] = self.moderate_post(url, client=client)
        return results


class AutomatedLabeler:
    """
    Assignment-required wrapper so that TA's test_labeler.py can
    call this class directly.

    This simply forwards calls to your PolicyProposalLabeler.
    """

    def __init__(self, client: Optional[Client] = None, input_dir: Optional[str] = None):
        # TA's grading script passes `client` and `input_dir`
        # Forward the provided authenticated client to the labeler.
        self.client = client
        self.input_dir = input_dir
        self.labeler = PolicyProposalLabeler(client=client)

    def moderate_post(self, url: str):
        return self.labeler.moderate_post(url)


# If someone runs this file directly, do a simple demo.
if __name__ == "__main__":
    labeler = PolicyProposalLabeler()
    test_url = "https://bsky.app/profile/example/post/12345"
    print(labeler.moderate_post(test_url))
