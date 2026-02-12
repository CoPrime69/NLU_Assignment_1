---
layout: default
title: Sports vs Politics Text Classification
---

# Sports vs Politics Text Classification (NLP Assignment)

## Project Objective
Build a binary text classifier that predicts whether an unseen text belongs to **Sports** or **Politics**, using classical machine learning with multiple feature representations.

## Dataset
This project uses the same raw source files as the reference dataset repository:
- `data/raw/politics.csv`
- `data/raw/sports.csv`

After cleaning and balancing:
- Total samples: **9186**
- Politics: **4593**
- Sports: **4593**
- Split: **80% train / 20% test** (stratified)

## Feature Representations
1. Bag of Words (`CountVectorizer`)
2. TF-IDF (`TfidfVectorizer`, 1-2 grams)
3. N-gram TF-IDF (`TfidfVectorizer`, 1-3 grams)

## ML Models Compared
1. Linear SVM
2. Logistic Regression
3. Multinomial Naive Bayes

Total experiments: **9 (3 × 3)**

## Results Summary
Best model: **logistic_regression__bow**

- Accuracy: **0.9587**
- Macro F1: **0.9586**
- Training time: **0.261 s**

Second-best: **linear_svm__tfidf** (Accuracy **0.9581**, Macro F1 **0.9581**)

### Best-model confusion matrix
- Politics → Politics: **876**
- Politics → Sports: **43**
- Sports → Politics: **33**
- Sports → Sports: **886**

## Key Insight
For this short-text dataset, simple unigram word-count features with a linear classifier gave the best trade-off between speed and accuracy.

## CSV Deliverables
- `results/all_results.csv`
- `results/per_class_results.csv`
- `results/confusion_matrices.csv`

## Reproducibility
Run locally:

```bash
python src/prepare_dataset.py
python src/train_models.py
```

## Report
Full report available at:
- `docs/report.md`

## Limitations
- Binary-only decision (no mixed-category output)
- No deep semantic understanding
- Single train-test split (no cross-validation in current submission)
- Potential domain drift over time
