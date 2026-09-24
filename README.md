# Cybersecurity Email Detection

Spam versus ham classifier for security triage. Paste one message or upload a mailbox, and the app returns a label plus a confidence score.

Applied AI / cybersecurity student project by Kadien Coe (KitchExtremeX).

## 60-second path

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train         # optional: outputs/ already contains a trained model
streamlit run app/streamlit_app.py
```

Open the local URL Streamlit prints. On **Single email**, choose **Load phishing example** or paste your own subject and body. The label and confidence show as soon as the text is committed.

Python 3.11 or newer. Run every command from the repository root so `src` imports resolve.

## Problem

Security teams still lose time on inbound mail: credential lures, fake invoices, package holds, and wire requests sit in the same queue as real project mail. This project is a small triage aid. It scores a message as **spam** or **ham** (legitimate) so a person can sort a mailbox faster. The score is a hint. It is a student portfolio model trained on synthetic mail, and a real deployment needs a licensed corpus and a fresh evaluation.

## Features

- Single-message scoring with class probabilities.
- Batch scoring of a Unix `.mbox` file, an on-screen table, and a CSV download.
- Training comparison of SVM, logistic regression, decision tree, and random forest on the same TF-IDF features.
- The saved model is the one with the highest spam-class F1.
- Timestamped training and inference logs in `logs/`.
- A committed model and vectorizer so the Streamlit app runs before you retrain.

## Project layout

```
data/emails.csv             # 1,600 labeled messages (text, label, category)
data/sample_inbox.mbox      # 8 synthetic messages for the batch demo
src/train/                  # load, vectorize, compare, save
src/predict/                # load artifacts, score text, parse mbox
src/preprocess.py           # shared normalizer stored inside the vectorizer
app/streamlit_app.py        # triage UI
outputs/                    # model.joblib, vectorizer.joblib, metrics.json
logs/                       # train.log and predict.log (gitignored)
scripts/generate_dataset.py # rebuild the synthetic corpus
scripts/train.py            # same entry as python -m src.train
tests/                      # unit and smoke tests
```

Training code and inference code live in separate packages. The UI calls `src.predict` only.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Runtime dependencies are scikit-learn, pandas, numpy, joblib, and Streamlit.

## Train

```bash
python -m src.train
```

Equivalent wrapper:

```bash
python scripts/train.py
```

Useful flags: `--data`, `--output-dir`, `--log-file`, `--test-size` (default `0.25`), `--seed` (default `42`).

The run prints a metric table and writes:

| File | Contents |
| --- | --- |
| `outputs/model.joblib` | Best estimator, refit on the full corpus after selection |
| `outputs/vectorizer.joblib` | TF-IDF vectorizer, same refit |
| `outputs/metrics.json` | Comparison, split, and the winning model name |
| `outputs/metrics.csv` | One row per model |
| `logs/train.log` | Timestamped copy of the same report |

If `outputs/model.joblib` or `outputs/vectorizer.joblib` is missing, prediction and Streamlit stop with: train first using `python -m src.train`.

## Predict

```bash
python -m src.predict --text "Subject: Q3 review

Hi team, the notes are on the shared drive."

python -m src.predict --mbox data/sample_inbox.mbox --csv /tmp/scores.csv
```

`--mbox` without `--csv` prints CSV to stdout. Columns are `subject`, `from`, `date`, `snippet`, `label`, `confidence`.

## Streamlit

```bash
streamlit run app/streamlit_app.py
```

- **Single email:** paste text. Start the text with `Subject:` so it matches the training layout. Example buttons load a legitimate note and a credential lure.
- **Batch mbox:** upload a `.mbox` file, or try `data/sample_inbox.mbox`. Download `spam_ham_classifications.csv`.
- The sidebar shows the loaded model name, the artifact directory, and the holdout spam F1. **Reload artifacts** picks up a retrain without restarting the process.

## Data schema and provenance

`data/emails.csv` is UTF-8. Required columns:

| Column | Required | Values |
| --- | --- | --- |
| `text` | yes | Full message. Rows in this repo start with `Subject:` and then the body. |
| `label` | yes | `spam` or `ham`. `1` / `0` are also accepted (`1` = spam). |
| `category` | no | Scenario name such as `credential_phish` or `meeting`. Used only to hold out whole scenarios during evaluation. It is not a model feature. |

The committed file has **1,600** rows: **800 spam** and **800 ham**, across 24 categories, generated with seed **42**.

Rebuild it with:

```bash
python scripts/generate_dataset.py
```

Every message is original synthetic text produced by that script. The corpus does not contain Enron mail, SpamAssassin public corpora, Ling-Spam, or the SMS Spam Collection. Addresses use the `.example` domain. Regenerate the CSV, then retrain, if you change the generator.

Text normalization (saved inside the vectorizer): lowercase, HTML stripped, URLs replaced with `url` plus hostname words, email addresses replaced with `emailaddr`, digits replaced with `num`. Features are TF-IDF word unigrams and bigrams.

## Model comparison

Spam is the positive class. Precision, recall, and F1 below are for spam. The winner is the highest spam F1. Ties break on macro F1, then accuracy, then alphabetical model name.

Evaluation holds out entire categories (about 25% of the categories for each label, seed 42). A category never appears in both train and test, so the score measures a new scenario rather than a repeated sentence. Held-out categories in the committed run: `bec_wire`, `code_review`, `fake_invoice`, `meeting`, `personal`, `prize_lottery` (1,200 train / 400 test).

A random row split on this corpus scores every model at F1 1.0 because the same templates appear on both sides. The category holdout is what makes the comparison visible.

| Model | Accuracy | Precision (spam) | Recall (spam) | F1 (spam) | F1 (macro) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **svm (saved)** | 0.9500 | 0.9945 | 0.9055 | **0.9479** | 0.9499 |
| logistic_regression | 0.9475 | 1.0000 | 0.8955 | 0.9449 | 0.9474 |
| random_forest | 0.9150 | 0.9941 | 0.8358 | 0.9081 | 0.9145 |
| decision_tree | 0.8875 | 0.9105 | 0.8607 | 0.8849 | 0.8874 |

The linear SVM is a calibrated `LinearSVC` so it can report probabilities. After the winner is chosen, that algorithm and the vectorizer are refit on all 1,600 rows and saved. Holdout numbers describe the selection split. They are not a second score of the refit artifact. Full precision lives in `outputs/metrics.json`.

These figures describe the synthetic holdout. A production triage model needs a current, properly licensed corpus of real mail and a new evaluation.

## Data license

The messages in `data/emails.csv` and `data/sample_inbox.mbox` are synthetic text written for this repository. You may reuse the corpus and `scripts/generate_dataset.py` for education and portfolio work. Present the rows as synthetic examples, not as real intercepted mail. No third-party email corpus is redistributed here.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite checks the schema, the category holdout, metric tie-breaks, mbox parsing, a short training run, and the committed model on obvious ham and spam text.
