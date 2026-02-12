import os
import time
import glob
import pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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


def get_param_grids():
    return {
        "linear_svm": {
            "model__C": [0.5, 1.0, 2.0],
            "model__class_weight": [None, "balanced"],
        },
        "logistic_regression": {
            "model__C": [0.5, 1.0, 2.0, 4.0],
            "model__class_weight": [None, "balanced"],
        },
        "multinomial_nb": {
            "model__alpha": [0.2, 0.4, 0.6, 0.8, 1.0],
        },
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
    complete_df = pd.read_csv(os.path.join(processed_dir, "complete_dataset.csv"))

    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]
    x_full = complete_df["text"]
    y_full = complete_df["label"]

    feature_extractors = get_feature_extractors()
    models = get_models()
    param_grids = get_param_grids()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for old_model in glob.glob(os.path.join(models_dir, "*.joblib")):
        os.remove(old_model)

    overall_rows = []
    per_class_rows = []
    confusion_rows = []
    cv_rows = []
    full_rows = []

    for feature_name, vectorizer in feature_extractors.items():
        for model_name, estimator in models.items():
            experiment = f"{model_name}__{feature_name}"
            print(f"Running {experiment}")

            pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", estimator),
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )

            start_train = time.time()
            grid.fit(x_train, y_train)
            tuning_time = time.time() - start_train

            best_pipeline = grid.best_estimator_
            best_params = grid.best_params_
            best_cv_score = float(grid.best_score_)

            cv_rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature_name,
                    "best_cv_f1_macro": best_cv_score,
                    "best_params": str(best_params),
                    "cv_tuning_time_seconds": tuning_time,
                }
            )

            start_pred = time.time()
            predictions = best_pipeline.predict(x_test)
            inference_time = time.time() - start_pred

            metrics = evaluate_predictions(y_test, predictions)

            overall_rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature_name,
                    "training_time_seconds": tuning_time,
                    "inference_time_seconds": inference_time,
                    "best_cv_f1_macro": best_cv_score,
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

            final_full_pipeline = Pipeline([
                ("vectorizer", vectorizer),
                ("model", estimator),
            ])
            final_full_pipeline.set_params(**best_params)

            full_start = time.time()
            final_full_pipeline.fit(x_full, y_full)
            full_train_time = time.time() - full_start

            model_path = os.path.join(models_dir, f"{experiment}.pkl")
            with open(model_path, "wb") as handle:
                pickle.dump(final_full_pipeline, handle)

            full_rows.append(
                {
                    "experiment": experiment,
                    "model": model_name,
                    "feature": feature_name,
                    "trained_on": "complete_dataset",
                    "full_dataset_size": int(len(complete_df)),
                    "full_training_time_seconds": full_train_time,
                    "model_path": model_path,
                }
            )

    overall_df = pd.DataFrame(overall_rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    per_class_df = pd.DataFrame(per_class_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    cv_df = pd.DataFrame(cv_rows).sort_values(["best_cv_f1_macro"], ascending=False)
    full_models_df = pd.DataFrame(full_rows)

    overall_df.to_csv(os.path.join(results_dir, "all_results.csv"), index=False)
    per_class_df.to_csv(os.path.join(results_dir, "per_class_results.csv"), index=False)
    confusion_df.to_csv(os.path.join(results_dir, "confusion_matrices.csv"), index=False)
    cv_df.to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)
    full_models_df.to_csv(os.path.join(results_dir, "full_data_models.csv"), index=False)

    best = overall_df.iloc[0]
    print("\nBest Experiment")
    print(f"{best['experiment']} | accuracy={best['accuracy']:.4f} | f1_macro={best['f1_macro']:.4f}")
    print("Saved models as .pkl trained on complete_dataset")
    print(f"Saved CSV files in: {results_dir}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_experiments(root)
