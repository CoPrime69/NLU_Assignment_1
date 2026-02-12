import os
import time
import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


RANDOM_STATE = 27


def get_feature_extractors():
    return {
        "bow": CountVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 1)),
        "tfidf": TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2), sublinear_tf=True),
        "ngram": TfidfVectorizer(max_features=18000, stop_words="english", ngram_range=(1, 3), min_df=2),
    }


def get_models():
    return {
        "linear_svm": LinearSVC(random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "multinomial_nb": MultinomialNB(alpha=0.6),
    }


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_experiments(project_root: str):
    processed_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    results_dir = os.path.join(project_root, "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]

    feature_extractors = get_feature_extractors()
    models = get_models()

    overall_rows = []
    per_class_rows = []
    confusion_rows = []

    for feature_name, vectorizer in feature_extractors.items():
        for model_name, estimator in models.items():
            experiment = f"{model_name}__{feature_name}"
            print(f"Running {experiment}")

            pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", estimator),
            ])

            start_train = time.time()
            pipeline.fit(x_train, y_train)
            train_time = time.time() - start_train

            start_pred = time.time()
            predictions = pipeline.predict(x_test)
            inference_time = time.time() - start_pred

            metrics = evaluate_predictions(y_test, predictions)

            overall_rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature_name,
                    "training_time_seconds": train_time,
                    "inference_time_seconds": inference_time,
                    **metrics,
                }
            )

            report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
            for class_label in ["Politics", "Sports"]:
                per_class_rows.append(
                    {
                        "experiment": experiment,
                        "model": model_name,
                        "feature": feature_name,
                        "class_label": class_label,
                        "precision": report[class_label]["precision"],
                        "recall": report[class_label]["recall"],
                        "f1_score": report[class_label]["f1-score"],
                        "support": int(report[class_label]["support"]),
                    }
                )

            cm = confusion_matrix(y_test, predictions, labels=["Politics", "Sports"])
            confusion_rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature_name,
                    "labels_order": "Politics,Sports",
                    "politics_pred_politics": int(cm[0, 0]),
                    "politics_pred_sports": int(cm[0, 1]),
                    "sports_pred_politics": int(cm[1, 0]),
                    "sports_pred_sports": int(cm[1, 1]),
                }
            )

            joblib.dump(pipeline, os.path.join(models_dir, f"{experiment}.joblib"))

    overall_df = pd.DataFrame(overall_rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    overall_df.to_csv(os.path.join(results_dir, "all_results.csv"), index=False)
    per_class_df.to_csv(os.path.join(results_dir, "per_class_results.csv"), index=False)
    confusion_df.to_csv(os.path.join(results_dir, "confusion_matrices.csv"), index=False)

    best = overall_df.iloc[0]
    print("\nBest Experiment")
    print(f"{best['experiment']} | accuracy={best['accuracy']:.4f} | f1_macro={best['f1_macro']:.4f}")
    print(f"Saved CSV files in: {results_dir}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_experiments(root)
