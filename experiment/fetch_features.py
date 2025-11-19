import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm
from urllib.parse import urlparse

API_BASE = "https://bsky.social/xrpc"

# =========================
# URL → at-uri 轉換
# =========================

def extract_uri_from_url(url: str) -> Optional[str]:
    """Convert Bluesky post URL to at:// URI.
    
    期望 URL path 類似：
      /profile/{handle}/post/{post_id}
    """
    if not isinstance(url, str) or "/post/" not in url:
        return None
    try:
        parts = urlparse(url).path.split("/")
        # ['', 'profile', '{handle}', 'post', '{id}']
        handle, post_id = parts[2], parts[4]
        return f"at://{handle}/app.bsky.feed.post/{post_id}"
    except Exception:
        return None

# =========================
# 登入 & API 基本工具
# =========================

def login(handle: str, app_password: str) -> str:
    """Login once and obtain JWT access token."""
    res = requests.post(
        f"{API_BASE}/com.atproto.server.createSession",
        json={"identifier": handle, "password": app_password},
        timeout=10,
    )
    res.raise_for_status()
    return res.json()["accessJwt"]


def fetch_json(endpoint: str, headers: dict, params: dict | None = None) -> Optional[dict]:
    """Generic GET helper with basic error handling."""
    try:
        r = requests.get(
            f"{API_BASE}/{endpoint}", headers=headers, params=params, timeout=15
        )
        if r.status_code == 200:
            return r.json()
        else:
            # 讓呼叫端決定怎麼處理錯誤碼
            return None
    except Exception:
        return None

# =========================
# Facets / Profile 解析
# =========================

def get_profile_handle(did: str, headers: dict) -> str:
    """Fetch user handle from DID."""
    if not did:
        return ""
    try:
        r = requests.get(
            f"{API_BASE}/app.bsky.actor.getProfile",
            headers=headers,
            params={"actor": did},
            timeout=10,
        )
        if r.status_code == 200:
            j = r.json()
            return j.get("handle", "") or ""
    except Exception:
        pass
    return ""


def extract_facets_info(facets: List[dict], headers: dict) -> Dict[str, str]:
    """Extract mentions_handle (by handle) & tags from facets."""
    mentions_handles: List[str] = []
    tags: List[str] = []

    for f in facets or []:
        for feat in f.get("features", []):
            ftype = feat.get("$type", "")
            # mention
            if ftype.endswith("mention"):
                did = feat.get("did", "")
                if did:
                    handle = get_profile_handle(did, headers)
                    if handle:
                        mentions_handles.append(handle)
            # tag
            elif ftype.endswith("tag"):
                tag = feat.get("tag", "")
                if tag:
                    tags.append(tag)

    return {
        "mentions_handle": ";".join(mentions_handles),
        "tags": ";".join(tags),
    }

# =========================
# Embed / Quote 解析
# =========================

def get_post_thread(uri: str, headers: dict, depth: int = 2) -> Optional[dict]:
    """Call app.bsky.feed.getPostThread."""
    return fetch_json(
        "app.bsky.feed.getPostThread",
        headers=headers,
        params={"uri": uri, "depth": depth},
    )


def extract_quoted_fields(post: dict, headers: dict) -> Dict[str, str]:
    """從主貼文的 embed 中抽出 quoted 貼文的 author_handle 與 text。"""
    quoted_author_handle = ""
    quoted_text = ""

    embed = post.get("embed") or {}
    q_uri: Optional[str] = None

    # case 1: embed.record
    if "record" in embed and isinstance(embed["record"], dict):
        q_uri = embed["record"].get("uri")
    # case 2: recordWithMedia
    elif embed.get("$type", "").endswith("recordWithMedia"):
        q_uri = embed.get("record", {}).get("uri")

    if not q_uri:
        return {
            "quoted_author_handle": "",
            "quoted_text": "",
        }

    q_data = get_post_thread(q_uri, headers)
    if not q_data:
        return {
            "quoted_author_handle": "",
            "quoted_text": "",
        }

    q_post = q_data.get("thread", {}).get("post", {}) or {}
    q_author = q_post.get("author", {}) or {}
    q_record = q_post.get("record", {}) or {}

    quoted_author_handle = q_author.get("handle", "") or ""
    quoted_text = q_record.get("text", "") or ""

    return {
        "quoted_author_handle": quoted_author_handle,
        "quoted_text": quoted_text,
    }

# =========================
# 抽取單一貼文 features
# =========================

def get_post_features(at_uri: str, headers: dict) -> Dict[str, str]:
    """
    給定 at-uri，抓取單一貼文與其必要 context，輸出符合 features.csv schema 所需欄位。
    """
    data = get_post_thread(at_uri, headers)
    if not data:
        raise RuntimeError("getPostThread returned no data")

    thread = data.get("thread", {}) or {}
    post = thread.get("post", {}) or {}
    if not post:
        raise RuntimeError("No 'post' field in thread data")

    # 主貼文作者 & 文字
    author = post.get("author", {}) or {}
    record = post.get("record", {}) or {}

    author_handle = author.get("handle", "") or ""
    text = record.get("text", "") or ""

    # facets -> mentions_handle, tags
    facets = record.get("facets", []) or []
    facet_info = extract_facets_info(facets, headers)

    # parent (reply 對象)
    parent_author_handle = ""
    parent_text = ""
    parent = thread.get("parent")
    if parent and isinstance(parent.get("post"), dict):
        p = parent["post"]
        pa = p.get("author", {}) or {}
        pr = p.get("record", {}) or {}
        parent_author_handle = pa.get("handle", "") or ""
        parent_text = pr.get("text", "") or ""

    # quoted 貼文
    quoted_info = extract_quoted_fields(post, headers)

    # 整理成輸出 dict（不含 url / label / status / error_msg）
    result = {
        "author_handle": author_handle,
        "text": text,
        "quoted_author_handle": quoted_info["quoted_author_handle"],
        "quoted_text": quoted_info["quoted_text"],
        "mentions_handle": facet_info["mentions_handle"],
        "parent_author_handle": parent_author_handle,
        "parent_text": parent_text,
        "tags": facet_info["tags"],
    }

    # 保證全部為字串（避免 NaN）
    for k, v in result.items():
        if v is None:
            result[k] = ""
        else:
            result[k] = str(v)

    return result

# =========================
# 主流程：ground_truth.csv → features.csv
# =========================

def process_ground_truth(
    input_csv: str,
    output_csv: str,
    handle: str,
    app_password: str,
    sleep_between: float = 0.2,
) -> None:
    """
    讀取 ground_truth.csv (url, label)，透過 Bluesky API 抽取 features，
    並輸出到 features.csv。

    output schema:
      url, label,
      author_handle, text,
      quoted_author_handle, quoted_text,
      mentions_handle,
      parent_author_handle, parent_text,
      tags,
      status, error_msg
    """
    df_in = pd.read_csv(input_csv)
    if "url" not in df_in.columns or "label" not in df_in.columns:
        raise ValueError("Input CSV must contain 'url' and 'label' columns.")

    print("[INFO] Logging in to Bluesky...")
    token = login(handle, app_password)
    headers = {"Authorization": f"Bearer {token}"}

    rows: List[Dict[str, Any]] = []
    continuous_error_count = 0

    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Fetching features"):
        url = str(row["url"])
        label = row["label"]

        out_row: Dict[str, Any] = {
            "url": url,
            "label": label,
            "author_handle": "",
            "text": "",
            "quoted_author_handle": "",
            "quoted_text": "",
            "mentions_handle": "",
            "parent_author_handle": "",
            "parent_text": "",
            "tags": "",
            "status": "fail",
            "error_msg": "",
        }

        try:
            at_uri = extract_uri_from_url(url)
            if not at_uri:
                raise ValueError("Invalid Bluesky URL format")

            features = get_post_features(at_uri, headers)

            for k in [
                "author_handle",
                "text",
                "quoted_author_handle",
                "quoted_text",
                "mentions_handle",
                "parent_author_handle",
                "parent_text",
                "tags",
            ]:
                out_row[k] = features.get(k, "")

            out_row["status"] = "success"
            out_row["error_msg"] = ""
            continuous_error_count = 0

        except Exception as e:
            err_str = str(e)
            out_row["status"] = "fail"
            out_row["error_msg"] = err_str
            continuous_error_count += 1
            print(f"[ERROR] idx={idx}, url={url}: {err_str}")

            # 簡單的 retry / 重新登入策略
            if continuous_error_count >= 5:
                print("[INFO] Too many continuous errors. Re-logging in...")
                try:
                    token = login(handle, app_password)
                    headers = {"Authorization": f"Bearer {token}"}
                    continuous_error_count = 0
                except Exception as login_err:
                    print(f"[FATAL] Re-login failed: {login_err}")
                    # 照樣繼續跑剩下的 row，只是很可能都 fail
        finally:
            rows.append(out_row)
            time.sleep(sleep_between)

    df_out = pd.DataFrame(rows)

    # 確保欄位順序
    column_order = [
        "url",
        "label",
        "author_handle",
        "text",
        "quoted_author_handle",
        "quoted_text",
        "mentions_handle",
        "parent_author_handle",
        "parent_text",
        "tags",
        "status",
        "error_msg",
    ]
    df_out = df_out[column_order]

    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved features to {output_csv}")


if __name__ == "__main__":
    # ==== 請在這裡設定你的參數 ====
    GROUND_TRUTH_CSV = "ground_truth.csv"
    OUTPUT_CSV = "features.csv"

    # 你的 labeler 帳號 & app password
    HANDLE = "4rows.bsky.social"
    APP_PASSWORD = "zycv-sxud-5646-wjbh"
    # =============================

    process_ground_truth(
        input_csv=GROUND_TRUTH_CSV,
        output_csv=OUTPUT_CSV,
        handle=HANDLE,
        app_password=APP_PASSWORD,
        sleep_between=0.2,
    )
