"""
build_context.py
-----------------
Utility for producing a clean contextual string from raw post features.

Follows the logic of your step2_build_prompt.py.
"""

from typing import Optional


def build_context_string(features: Optional[dict]) -> str:
    """
    Build a single context string used for toxicity / attack reasoning.

    features fields expected:
        - text
        - quoted_text
        - quoted_author
        - parent_text
        - parent_author
        - mentions
        - tags
    """

    features = features or {}

    def _safe_text(value):
        if value is None:
            return ""
        return str(value)

    text = _safe_text(features.get("text", ""))

    quoted_author = _safe_text(features.get("quoted_author", ""))
    quoted_text = _safe_text(features.get("quoted_text", ""))

    parent_author = _safe_text(features.get("parent_author", ""))
    parent_text = _safe_text(features.get("parent_text", ""))

    mentions = features.get("mentions") or []
    tags = features.get("tags") or []

    if not isinstance(mentions, (list, tuple, set)):
        mentions = [mentions] if mentions else []
    if not isinstance(tags, (list, tuple, set)):
        tags = [tags] if tags else []

    mentions = [str(m) for m in mentions if m]
    tags = [str(t) for t in tags if t]

    # ---- Build readable context ----
    context_parts = []

    context_parts.append(f"Original post: {text}")

    if quoted_text:
        context_parts.append(f"Quoted from {quoted_author}: {quoted_text}")

    if parent_text:
        context_parts.append(f"Replying to {parent_author}: {parent_text}")

    if mentions:
        mention_str = ", ".join(mentions)
        context_parts.append(f"Mentions: {mention_str}")

    if tags:
        tag_str = ", ".join(tags)
        context_parts.append(f"Tags: {tag_str}")

    context_string = "\n".join(context_parts).strip()

    # Always return a non-empty string so downstream components are safe.
    return context_string or "Original post: "
