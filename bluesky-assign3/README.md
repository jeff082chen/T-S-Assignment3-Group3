# Bluesky Automated Labeler

This repository contains the code we used for CS 5342 Assignment 3. It builds a Bluesky automated labeler that fetches posts, runs a multi‑model pipeline (translation, toxicity/profanity, and targeted‑attack classifiers), and can optionally emit labels back to Bluesky.

## Prerequisites

- Python 3.10 or 3.11
- Git
- A Bluesky account and an **app password** (do not use your main login)
- ~3–4 GB of disk for local HuggingFace models stored under `models/`

## Clone and environment setup

1. Clone and enter the project (replace `<repo>` with your clone URL):

```bash
git clone <repo>
cd <repo>
```

2. Create a virtual environment and install dependencies:

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

3. Move to project directory:

```bash
cd ./bluesky-assign3
```

## Credentials

1. Copy the template and fill in your Bluesky handle and app password:

```bash
cp .env-TEMPLATE .env
# Open .env and set:
# USERNAME = "<your_handle.bsky.social>"
# PW = "<your_app_password>"
```

## Structure

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

## Running the test harness

The provided script checks that the labeler returns expected outputs for a CSV of URLs.

```bash
python test_labeler.py test-data test-data/data.csv
```

- The script logs any mismatches, then prints accuracy.

## Project layout

- `pylabel/policy_proposal_labeler.py` – entry point class used by graders/tests.
- `pp_labeler/` – feature fetching, context builder, and model inference utilities.
- `models/` – local caches of the HuggingFace models.
- `test_labeler.py` – harness for CSV-based evaluation and optional label emission.

## Targeted Profanity & Attack Detection Policy

_(Used for LLaMA moderation + distilled classifier)_

This project applies a strict, context-aware policy to classify social media posts into:

- **Neutral**
- **Profanity**
- **Targeted Attack**

The goal is to reliably distinguish **general profanity** from **directed hostility**, even in multilingual or context-rich posts.

## What Counts as “Profanity”

“Profanity” refers **strictly** to:

- Explicit swear words
- Racial/ethnic slurs
- Sexually explicit insults
- Aggressive derogatory expressions
- Violent language
- Threat-like wording (only considered an _attack_ when directed at a target)

**Profanity alone ≠ attack.**

It becomes an attack **only when directed at a target with hostile intent**.

## What Counts as a “Targeted Attack”

A post is a **targeted attack** when profanity is used to:

- **Insult**
- **Degrade**
- **Mock**
- **Threaten**

…a **specific target**, based on contextual cues such as @mentions, replies, quoted posts, named individuals, or proper nouns.

## Critical Exclusions (Not Attacks)

These are important exceptions: profanity that is _not hostile_.

### 1. Positive/Excited Sentiment (Intensifiers)

Profanity used to emphasize excitement or approval.

- “This is **fucking amazing**.”
- “Damn, that was **so good**.”
- “You **killed it**!” (positive idiom)

**Not an attack**

### 2. Friendly Banter / Community Slang

Profanity used affectionately within positive context.

- “Love you, you crazy **mofos**.”
- “She’s a **bad bitch** (compliment).”

**Not an attack**

### 3. Non-violent Figurative Language

Metaphor, exaggeration, or idioms.

- “I’m going to **destroy you** in this game.”
- “I could **die** of embarrassment.”
- “This song **slaps**.”

**Not an attack**

### 4. Non-hostile Regional Slang

Mild expletives without hostility.

- “That was **bloody** awful.”
- “This thing is a **bloody** mess.”

**Not an attack**

### 5. Proper Nouns That Contain Sensitive Words

Words that _look_ profane but aren’t.

- “Punta Cana”
- “Lil Darkie” (artist name)

**Not an attack**

### 6. Emotional but Non-directed Negativity

No target + no profanity = not an attack.

- “I’m **so tired** of everything.”
- “This day sucks.”

**Not an attack**

## Attack Categories (With Clear Examples)

A post becomes an **attack** only when profanity is **targeted**.
The system uses 5 attack types:

### 1. Attack by Quote

Profanity targets the **author of a quoted post**.

**Example (attack):**

> “This idiot in the quote is a total **asshole**.”

### 2. Attack by Mention

Profanity targets an @mentioned user.

**Example (attack):**

> “@user you sound like a **dumb piece of shit**.”

### 3. Attack by Reply

Profanity targets the author of the replied-to post.

**Example (attack):**

> “You clearly have no brain. What a **moron**.”

### 4. Attack Toward a Named Person

Profanity directed at an identifiable individual.

**Example (attack):**

> “Elon Musk is a clueless **idiot**.”

### 5. Attack Toward a Proper Noun

Profane hostility aimed at groups, organizations, or entities.

**Example (attack):**

> “Your whole company is full of **scumbags**.”

**Example (attack):**

> “Those **idiot** politicians don’t know anything.”

## Edge Cases

These are common tricky examples and how the system labels them.

### Not Attacks

| Example                                                          | Reason                                                     |
| ---------------------------------------------------------------- | ---------------------------------------------------------- |
| “That was bloody stupid.”                                        | British intensifier, not violent                           |
| “You’re crazy lol.”                                              | Not hostile                                                |
| “You’re annoying as hell.”                                       | No profanity + no threat                                   |
| “Kill yourself” (as slang in gaming, _if clearly joking/banter_) | Still **profanity**, but only attack if target + hostility |
| “I want to punch this damn wall.”                                | Self-directed, not attack                                  |

### Attacks

| Example                     | Reason               |
| --------------------------- | -------------------- |
| “I’ll beat your ass.”       | Threat + target      |
| “@user shut up you bitch”   | Mention + profanity  |
| “Those idiots in Congress…” | Proper noun targeted |
