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

Structure
-----------
```bash
bluesky-assign3
├── models                               # HuggingFace model caches
│   ├── adult-bert
│   ├── attack-classifier
│   ├── nllb
│   └── toxic-bert
├── pp_labeler                           # Labeler utilities
│   ├── __init__.py 
│   ├── build_context.py                 # Biuld input context for models
│   ├── fetch_features.py                # Fetch post data from Bluesky
│   └── model_inference.py               # Run model inference
├── pylabel                              # Labeler entry point
│   ├── __init__.py
│   ├── label.py                         # Labeler base class
│   └── policy_proposal_labeler.py       # Main labeler class
├── test-data                            # Ground truth test data
│   └── data.csv
└── test_labeler.py                      # Main test script
```

Running the test harness
------------------------
The provided script checks that the labeler returns expected outputs for a CSV of URLs.
```bash
python test_labeler.py test-data test-data/data.csv
```
- The script logs any mismatches, then prints accuracy.

Project layout
--------------
- `pylabel/policy_proposal_labeler.py` – entry point class used by graders/tests.
- `pp_labeler/` – feature fetching, context builder, and model inference utilities.
- `models/` – local caches of the HuggingFace models.
- `test_labeler.py` – harness for CSV-based evaluation and optional label emission.
