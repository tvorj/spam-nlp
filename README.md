# SMS Spam Detection (NLP) — TF-IDF + Logistic Regression

A small end-to-end NLP project that classifies SMS messages as **spam** or **ham** using a scikit-learn pipeline.

## What this notebook does
The notebook `01_spam_baseline.ipynb`:
1. Loads the dataset from a local TSV-like file.
2. Cleans the label column (`"ham` → `ham`, `"spam` → `spam`).
3. Splits the data into train/validation sets (stratified).
4. Builds an NLP baseline with `TfidfVectorizer` + `LogisticRegression` inside an `sklearn.Pipeline`.
5. Evaluates the model with Accuracy / F1 / ROC-AUC + confusion matrix.
6. Computes Precision/Recall for the **spam** class.
7. Displays examples of false negatives / false positives.
8. Prints the top tokens that push predictions toward spam vs ham.

## Dataset
- File: `../data/SMSSpamCollection.csv`
- Format: 2 columns (label, text) separated by TAB.
- Columns used:
  - `label`: `ham` / `spam`
  - `text`: message text

> The dataset is expected locally in `data/`. (Recommended: do not commit it to GitHub.)

## Modeling approach
- **Split**: `train_test_split(test_size=0.2, random_state=45, stratify=y)`
- **Vectorization**: `TfidfVectorizer`
  - `lowercase=True`
  - `stop_words="english"`
  - `ngram_range=(1, 2)` (unigrams + bigrams)
  - `min_df=2` (drop extremely rare tokens)
- **Classifier**: `LogisticRegression(max_iter=2000)`
- **Why Pipeline**: keeps preprocessing + model together and makes the workflow reproducible.

## Results (validation split)
- **Accuracy:** 0.9659  
- **F1 (spam as positive):** 0.8550  
- **ROC-AUC:** 0.9941  
- **Precision (spam):** 0.9912  
- **Recall (spam):** 0.7517  

Confusion matrix counts (val):
- Ham → Ham: 965
- Ham → Spam (FP): 1
- Spam → Ham (FN): 37
- Spam → Spam: 112

## Token analysis (interpretability)
The notebook prints top features (tokens) driving predictions.
Examples:
- **Spam indicators:** `free`, `claim`, `urgent`, `prize`, `cash`, `win`, `reply`, `txt`, `mobile`, ...
- **Ham indicators:** `ok`, `come`, `home`, `sorry`, `going`, `good`, ...

## How to run

### 1) Create env and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
