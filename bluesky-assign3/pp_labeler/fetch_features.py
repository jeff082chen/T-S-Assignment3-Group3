"""
fetch_features.py
------------------
Fetch Bluesky post features for a *single* URL.

This wraps the logic from your existing step1_fetch_features.py,
but restructures it so policy_proposal_labeler.py can call it directly.
"""

from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from atproto import Client


# ------------------------------------------------------------------------------
# Helper: extract record_key & handle from Bluesky URL
# ------------------------------------------------------------------------------
def parse_bsky_url(url: str) -> Tuple[str, str]:
    """
    Convert Bluesky URL variants like:
        https://bsky.app/profile/<handle>/post/<rkey>
    into (handle, rkey).

    Supports staging subdomains and DID-based handles.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Bluesky URL must be a non-empty string.")

    parsed = urlparse(url.strip())
    parts = [segment for segment in parsed.path.split("/") if segment]

    handle = None
    rkey = None
    for idx, part in enumerate(parts):
        if part == "profile" and idx + 1 < len(parts):
            handle = parts[idx + 1]
        if part == "post" and idx + 1 < len(parts):
            rkey = parts[idx + 1]

    if not handle or not rkey:
        raise ValueError(f"Invalid Bluesky post URL: {url}")

    return handle, rkey

def did_to_handle(client, did: str) -> str:
    """
    Convert a DID to its handle using atproto client.

    Fallback: return DID if lookup fails.
    """
    if not did:
        return ""
    try:
        prof = client.get_profile(did)
        return getattr(prof, "handle", None) or did
    except Exception:
        return did

# ------------------------------------------------------------------------------
# Main single-post fetcher (for assignment)
# ------------------------------------------------------------------------------
def fetch_single_post_features(url: str, client: Client) -> Dict[str, object]:
    """
    Fetches contextual features for ONE Bluesky post using the provided client.

    Args:
        url (str): Bluesky post URL.
        client (Client): Logged-in Bluesky client supplied by the pipeline.

    Returns a dictionary with fields required by the pipeline:
        {
            "url": ...,
            "author_handle": ...,
            "text": ...,
            "quoted_author": ...,
            "quoted_text": ...,
            "parent_author": ...,
            "parent_text": ...,
            "mentions": [...],
            "tags": [...]
        }
    """
    if client is None:
        raise ValueError("fetch_single_post_features requires a logged-in client.")

    handle, rkey = parse_bsky_url(url)

    # Determine the canonical URI for the post so we can fetch the entire thread.
    thread = None
    uri_candidates: List[str] = []
    post_author_handle = ""

    try:
        post = client.get_post(rkey, handle)
        post_author_handle = getattr(post.author, "handle", "") or getattr(post.author, "did", "")
        if getattr(post, "uri", None):
            uri_candidates.append(post.uri)
        author_did = getattr(post.author, "did", None)
        if author_did:
            uri_candidates.append(f"at://{author_did}/app.bsky.feed.post/{rkey}")
    except Exception:
        post = None

    if handle.startswith("did:"):
        uri_candidates.append(f"at://{handle}/app.bsky.feed.post/{rkey}")
    else:
        uri_candidates.append(f"at://{handle}/app.bsky.feed.post/{rkey}")

    last_err: Optional[Exception] = None
    for candidate in uri_candidates:
        if not candidate:
            continue
        try:
            thread = client.app.bsky.feed.get_post_thread({"uri": candidate})
            break
        except Exception as exc:
            last_err = exc

    if thread is None:
        raise RuntimeError(f"Unable to fetch thread for {url}: {last_err}")

    thread_view = getattr(thread, "thread", None)
    post_view = getattr(thread_view, "post", None) if thread_view else None
    if post_view is None:
        raise RuntimeError(f"Thread payload missing post for {url}")

    record = getattr(post_view, "record", None)

    def safe_text(value) -> str:
        if not value:
            return ""
        if isinstance(value, str):
            return value
        return getattr(value, "text", "") or ""

    author_handle = getattr(post_view.author, "handle", "") if hasattr(post_view, "author") else ""
    if not author_handle and hasattr(post_view, "author"):
        author_handle = getattr(post_view.author, "did", "") or post_author_handle

    text = safe_text(record)

    # -------------------------
    # quoted post (if exists)
    # -------------------------
    quoted_author = ""
    quoted_text = ""
    embed = getattr(post_view, "embed", None)
    if embed is not None and hasattr(embed, "record"):
        try:
            rec = embed.record
            if rec and hasattr(rec, "author"):
                quoted_author = getattr(rec.author, "handle", "") or getattr(rec.author, "did", "")
            if rec and hasattr(rec, "value"):
                quoted_text = safe_text(rec.value)
        except Exception:
            pass

    # -------------------------
    # parent post (reply context)
    # -------------------------
    parent_author = ""
    parent_text = ""

    parent_node = getattr(thread_view, "parent", None) if thread_view else None
    if parent_node and hasattr(parent_node, "post"):
        try:
            parent_post = parent_node.post
            if hasattr(parent_post, "author"):
                parent_author = getattr(parent_post.author, "handle", "") or getattr(parent_post.author, "did", "")
            if hasattr(parent_post, "record"):
                parent_text = safe_text(parent_post.record)
        except Exception:
            pass

    # -------------------------
    # mentions & tags
    # -------------------------
    mentions: List[str] = []
    tags: List[str] = []

    mentions: List[str] = []
    facets = getattr(record, "facets", None) or []
    for facet in facets:
        features = getattr(facet, "features", None) or []
        for feature in features:
            did = getattr(feature, "did", None)
            if did:
                handle = did_to_handle(client, did)
                mentions.append(handle)
            tag = getattr(feature, "tag", None)
            if tag:
                tags.append(tag)

    record_tags = getattr(record, "tags", None) or []
    for tag in record_tags:
        tags.append(tag)

    return {
        "url": url,
        "author_handle": author_handle,
        "text": text,
        "quoted_author": quoted_author,
        "quoted_text": quoted_text,
        "parent_author": parent_author,
        "parent_text": parent_text,
        "mentions": mentions,
        "tags": list(tags),
    }
