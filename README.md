# Sports vs Politics Text Classifier (Fresh Project)

This is a from-scratch NLP assignment project that classifies text documents into **Sports** or **Politics** using classical machine learning.

## What is included
- Same raw dataset source files used in the original repository:
  - `data/raw/politics.csv`
  - `data/raw/sports.csv`
- New preprocessing and training pipeline implemented independently in this project.
- Three feature representations:
  - Bag of Words (`bow`)
  - TF-IDF (`tfidf`)
  - N-gram TF-IDF (`ngram`)
- Three ML models:
  - Linear SVM
  - Logistic Regression
  - Multinomial Naive Bayes
- CSV deliverables for quantitative comparison:
  - `results/all_results.csv`
  - `results/per_class_results.csv`
  - `results/confusion_matrices.csv`

## Final result snapshot
Best experiment on the generated test split:
- `logistic_regression__bow`
- Accuracy: `0.9587`
- Macro F1: `0.9586`

## Project structure
- `src/prepare_dataset.py`: data cleaning, balancing, split, summary
- `src/train_models.py`: 9 experiments (3 features × 3 models), metrics, model export
- `data/processed/`: processed train/test/full datasets
- `models/`: serialized trained pipelines (`*.joblib`)
- `results/`: all evaluation CSV files
- `docs/report.md`: detailed assignment report
- `site/index.md`: GitHub Pages content

## Run
See `QUICKSTART.md`.
