Bluesky Automated Labeler
=========================

This repository contains the code we used for CS 5342 Assignment 3. It builds a Bluesky automated labeler that fetches posts, runs a multi‑model pipeline (translation, toxicity/profanity, and targeted‑attack classifiers), and can optionally emit labels back to Bluesky.

Prerequisites
-------------
- Python 3.10 or 3.11
- Git
- A Bluesky account and an **app password** (do not use your main login)
- ~3–4 GB of disk for local HuggingFace models stored under `models/`

Clone and environment setup
---------------------------
1) Clone and enter the project (replace `<repo>` with your clone URL):
```bash
git clone <repo>
cd <repo>
```
2) Create a virtual environment and install dependencies:
```bash
# with uv
uv venv
source .venv/bin/activate
uv sync

# with pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Move to project directory:
```bash
cd ./bluesky-assign3
```

Credentials
-----------
1) Copy the template and fill in your Bluesky handle and app password:
```bash
cp .env-TEMPLATE .env
# Open .env and set:
# USERNAME = "<your_handle.bsky.social>"
# PW = "<your_app_password>"
```

Models
------
Pre-fetched models live in `models/` (`nllb`, `toxic-bert`, `adult-bert`, `attack-classifier`). If you remove them, the code will download from HuggingFace on first run (needs network access and disk space).

Running the test harness
------------------------
The provided script checks that the labeler returns expected outputs for a CSV of URLs.
```bash
python test_labeler.py test-data test-data/data.csv
```
- The script logs any mismatches, then prints accuracy.

Emit labels to Bluesky (optional)
---------------------------------
If you want to actually apply labels via your labeler account, add `--emit_labels`:
```bash
python test_labeler.py <labeler_inputs_dir> test-data/data.csv --emit_labels
```
This uses the credentials in `.env` to log in, proxy as a labeler, and emit labels for any non-empty predictions.

Optional threshold debugging
----------------------------
To recompute evaluation metrics for the tuned thresholds:
```bash
python debug_thresholds.py
python threshold_eval.py debug_metrics.csv
```
This prints accuracy plus precision/recall splits for neutral vs toxic and targeted vs profanity.

Project layout
--------------
- `pylabel/policy_proposal_labeler.py` – entry point class used by graders/tests.
- `pp_labeler/` – feature fetching, context builder, and model inference utilities.
- `models/` – local caches of the HuggingFace models.
- `test_labeler.py` – harness for CSV-based evaluation and optional label emission.
