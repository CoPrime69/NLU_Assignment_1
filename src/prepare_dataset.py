import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 27


def normalize_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dataset(project_root: str) -> None:
    raw_dir = os.path.join(project_root, "data", "raw")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    politics_path = os.path.join(raw_dir, "politics.csv")
    sports_path = os.path.join(raw_dir, "sports.csv")

    politics_df = pd.read_csv(politics_path)
    sports_df = pd.read_csv(sports_path)

    politics_df = politics_df[["text"]].copy()
    sports_df = sports_df[["text"]].copy()

    politics_df["label"] = "Politics"
    sports_df["label"] = "Sports"

    combined = pd.concat([politics_df, sports_df], ignore_index=True)
    combined["text"] = combined["text"].map(normalize_text)
    combined = combined.dropna(subset=["text"])
    combined = combined[combined["text"] != ""]
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    class_counts = combined["label"].value_counts()
    min_count = int(class_counts.min())

    politics_balanced = combined[combined["label"] == "Politics"].sample(
        n=min_count, random_state=RANDOM_STATE
    )
    sports_balanced = combined[combined["label"] == "Sports"].sample(
        n=min_count, random_state=RANDOM_STATE
    )

    balanced = (
        pd.concat([politics_balanced, sports_balanced], ignore_index=True)
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    train_df, test_df = train_test_split(
        balanced,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=balanced["label"],
    )

    balanced_path = os.path.join(processed_dir, "complete_dataset.csv")
    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    balanced.to_csv(balanced_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    lengths = balanced["text"].str.split().str.len()
    summary = pd.DataFrame(
        [
            {
                "total_samples": int(len(balanced)),
                "politics_samples": int((balanced["label"] == "Politics").sum()),
                "sports_samples": int((balanced["label"] == "Sports").sum()),
                "avg_tokens": float(lengths.mean()),
                "median_tokens": float(lengths.median()),
                "min_tokens": int(lengths.min()),
                "max_tokens": int(lengths.max()),
            }
        ]
    )

    summary_path = os.path.join(processed_dir, "dataset_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {balanced_path}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print(f"Saved: {summary_path}")
    print("Class distribution:")
    print(balanced["label"].value_counts())


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dataset(root)
