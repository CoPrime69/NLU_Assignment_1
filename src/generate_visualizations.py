import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="ticks", context="talk")
plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#fcfcfc",
        "axes.edgecolor": "#2f2f2f",
        "axes.titleweight": "bold",
        "axes.labelweight": "semibold",
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
    }
)

FEATURE_COLORS = {
    "bow": "#4C78A8",
    "tfidf": "#F58518",
    "ngram": "#54A24B",
}

MODEL_MARKERS = {
    "linear_svm": "o",
    "logistic_regression": "s",
    "multinomial_nb": "D",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(project_root: str):
    results_dir = os.path.join(project_root, "results")
    overall = pd.read_csv(os.path.join(results_dir, "all_results.csv"))
    per_class = pd.read_csv(os.path.join(results_dir, "per_class_results.csv"))
    confusion = pd.read_csv(os.path.join(results_dir, "confusion_matrices.csv"))
    return overall, per_class, confusion


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)


def plot_top_f1(overall: pd.DataFrame, out_dir: str):
    top = overall.sort_values("f1_macro", ascending=True).copy()
    plt.figure(figsize=(15, 8))
    ax = sns.barplot(
        data=top,
        y="experiment",
        x="f1_macro",
        hue="feature",
        dodge=False,
        palette=FEATURE_COLORS,
    )
    ax.set_title("Macro F1 Score by Experiment", pad=14)
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Experiment")
    ax.set_xlim(0.93, 0.965)
    style_axes(ax)

    for patch in ax.patches:
        value = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(value + 0.0003, y, f"{value:.4f}", va="center", fontsize=10)

    ax.legend(title="Feature", loc="lower right")
    plt.savefig(os.path.join(out_dir, "f1_by_experiment.png"), dpi=220)
    plt.close()


def plot_accuracy_vs_train_time(overall: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=overall,
        x="training_time_seconds",
        y="accuracy",
        hue="feature",
        style="model",
        markers=MODEL_MARKERS,
        s=220,
        palette=FEATURE_COLORS,
        edgecolor="black",
        linewidth=0.8,
    )

    for _, row in overall.iterrows():
        label = f"{row['model']}\n{row['feature']}"
        ax.annotate(
            label,
            (row["training_time_seconds"], row["accuracy"]),
            xytext=(8, 7),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bbbbbb", alpha=0.8),
        )

    ax.set_title("Accuracy vs Training Time Trade-off", pad=12)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.942, 0.962)
    style_axes(ax)
    ax.legend(title="Feature / Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_training_time.png"), dpi=220)
    plt.close()


def plot_feature_average(overall: pd.DataFrame, out_dir: str):
    feature_scores = (
        overall.groupby("feature", as_index=False)[["accuracy", "f1_macro", "precision_macro", "recall_macro"]]
        .mean()
        .melt(id_vars="feature", var_name="metric", value_name="score")
    )

    metric_order = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    feature_scores["metric"] = pd.Categorical(feature_scores["metric"], categories=metric_order, ordered=True)

    plt.figure(figsize=(11, 7))
    ax = sns.barplot(
        data=feature_scores,
        x="feature",
        y="score",
        hue="metric",
        palette="tab10",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Average Metric by Feature Representation", pad=12)
    ax.set_xlabel("Feature Type")
    ax.set_ylabel("Score")
    ax.set_ylim(0.93, 0.965)
    style_axes(ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    ax.legend(title="Metric", ncol=2, loc="upper center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_metric_comparison.png"), dpi=220)
    plt.close()


def plot_iteration_curve(overall: pd.DataFrame, out_dir: str):
    ordered = overall.sort_values(["feature", "model"]).reset_index(drop=True).copy()
    ordered["iteration"] = ordered.index + 1

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=ordered,
        x="iteration",
        y="f1_macro",
        hue="feature",
        style="model",
        markers=True,
        dashes=False,
        linewidth=2.2,
        markersize=9,
        palette=FEATURE_COLORS,
    )

    best_idx = ordered["f1_macro"].idxmax()
    best_row = ordered.loc[best_idx]
    ax.scatter(best_row["iteration"], best_row["f1_macro"], s=300, c="#E45756", marker="*", zorder=5)
    ax.annotate(
        f"Best: {best_row['experiment']}\nF1={best_row['f1_macro']:.4f}",
        (best_row["iteration"], best_row["f1_macro"]),
        xytext=(14, -20),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9),
    )

    ax.set_title("Iteration-wise Macro F1 Trend", pad=12)
    ax.set_xlabel("Iteration Number")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(ordered["iteration"])
    ax.set_ylim(0.942, 0.962)
    ax.tick_params(axis="x", rotation=0)
    style_axes(ax)
    ax.legend(title="Feature / Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iteration_f1_curve.png"), dpi=220)
    plt.close()


def plot_best_confusion_heatmap(overall: pd.DataFrame, confusion: pd.DataFrame, out_dir: str):
    best_experiment = overall.sort_values("f1_macro", ascending=False).iloc[0]["experiment"]
    row = confusion[confusion["experiment"] == best_experiment].iloc[0]

    counts_matrix = [
        [row["politics_pred_politics"], row["politics_pred_sports"]],
        [row["sports_pred_politics"], row["sports_pred_sports"]],
    ]
    counts_df = pd.DataFrame(
        counts_matrix,
        index=["True Politics", "True Sports"],
        columns=["Pred Politics", "Pred Sports"],
    )
    norm_df = counts_df.div(counts_df.sum(axis=1), axis=0)

    annot = counts_df.astype(str) + "\n(" + (norm_df * 100).round(1).astype(str) + "%)"

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        norm_df,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=1.2,
        linecolor="white",
        cbar_kws={"label": "Row-wise proportion"},
    )
    ax.set_title(f"Best Model Confusion Matrix\n{best_experiment}", pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_confusion_matrix.png"), dpi=240)
    plt.close()


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "results", "visualizations")
    ensure_dir(out_dir)

    overall, _, confusion = load_data(project_root)

    plot_top_f1(overall, out_dir)
    plot_accuracy_vs_train_time(overall, out_dir)
    plot_feature_average(overall, out_dir)
    plot_iteration_curve(overall, out_dir)
    plot_best_confusion_heatmap(overall, confusion, out_dir)

    print(f"Saved visualizations in: {out_dir}")


if __name__ == "__main__":
    main()
