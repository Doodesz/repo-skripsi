"""Train the four fourth-try STAI-labelled WESAD models.

The reference-compatible split is a random 80/20 window split with seed 0,
matching the forked notebook. Grouped evaluation is reported separately so
overlapping windows from one subject cannot cross the evaluation boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


REPO_ROOT = next(
    path for path in [Path.cwd(), *Path.cwd().parents] if (path / "WESAD").exists()
)
DATA_DIR = REPO_ROOT / "Data" / "Preprocessed n STAI-labeled 6s"
MODEL_DIR = REPO_ROOT / "Models" / "stai_6s"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "binary_anxiety"
GROUP = "subject_id"
METADATA_COLUMNS = {
    TARGET,
    GROUP,
    "condition",
    "window_start_seconds",
    "window_end_seconds",
    "condition_stai_mean",
    "baseline_stai_mean",
}

RANDOM_STATE = 0
TEST_SIZE = 0.2


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(DATA_DIR.glob("S*_stai_60s_shift6s.pkl"))
    if not files:
        raise FileNotFoundError(
            f"No fourth-try datasets found in {DATA_DIR}. "
            "Run stai_6s_pipeline.py first."
        )

    frames = []
    for path in files:
        frame = pd.read_pickle(path)
        required = {TARGET, GROUP, *METADATA_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[data[TARGET].isin([0, 1])].copy()
    feature_columns = [
        column for column in data.columns if column not in METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after removing metadata.")

    numeric = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    data.loc[:, feature_columns] = numeric.replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    if data[TARGET].nunique() != 2:
        raise ValueError(
            f"Both binary classes are required; got "
            f"{data[TARGET].value_counts().to_dict()}"
        )
    return data, feature_columns


def build_models() -> dict[str, object]:
    # RF and linear SVM match the explicitly recorded forked configuration.
    # XGBoost and KNN use the existing project configuration because the
    # forked notebook does not define those classifiers.
    return {
        "random_forest": RandomForestClassifier(
            max_depth=4, random_state=RANDOM_STATE
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1, random_state=RANDOM_STATE),
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "knn": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
    }


def evaluate_split(
    data: pd.DataFrame,
    feature_columns: list[str],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split_name: str,
) -> dict[str, object]:
    X_train = data.iloc[train_indices][feature_columns]
    X_test = data.iloc[test_indices][feature_columns]
    y_train = data.iloc[train_indices][TARGET].astype(int)
    y_test = data.iloc[test_indices][TARGET].astype(int)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            f"{split_name} must contain both classes in train and test; "
            f"got train={y_train.value_counts().to_dict()}, "
            f"test={y_test.value_counts().to_dict()}"
        )

    results: dict[str, object] = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "split": split_name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["No Anxiety", "Anxiety"],
                output_dict=True,
                zero_division=0,
            ),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_subjects": sorted(
                data.iloc[train_indices][GROUP].astype(int).unique().tolist()
            ),
            "test_subjects": sorted(
                data.iloc[test_indices][GROUP].astype(int).unique().tolist()
            ),
        }
        joblib.dump(model, MODEL_DIR / f"{name}_{split_name}.joblib")
    return results


def evaluate_loso(
    data: pd.DataFrame, feature_columns: list[str]
) -> dict[str, object]:
    """Evaluate every subject once as an unseen test subject."""
    indices = np.arange(len(data))
    y = data[TARGET].astype(int).to_numpy()
    groups = data[GROUP].astype(int).to_numpy()
    predictions_by_model: dict[str, np.ndarray] = {
        name: np.full(len(data), -1, dtype=int) for name in build_models()
    }
    fold_subjects: list[int] = []

    for train_indices, test_indices in LeaveOneGroupOut().split(
        indices, y, groups=groups
    ):
        test_subject = int(data.iloc[test_indices][GROUP].iloc[0])
        fold_subjects.append(test_subject)
        if y[train_indices].min() == y[train_indices].max():
            raise ValueError(
                f"LOSO training fold for subject S{test_subject} contains one class."
            )
        for name, model in build_models().items():
            model.fit(
                data.iloc[train_indices][feature_columns],
                data.iloc[train_indices][TARGET].astype(int),
            )
            predictions_by_model[name][test_indices] = model.predict(
                data.iloc[test_indices][feature_columns]
            )

    results: dict[str, object] = {
        "split": "leave_one_subject_out",
        "subjects_evaluated": sorted(fold_subjects),
    }
    for name, predictions in predictions_by_model.items():
        if (predictions < 0).any():
            raise RuntimeError(f"LOSO did not produce predictions for {name}.")
        subject_metrics = []
        for subject in sorted(np.unique(groups)):
            subject_mask = groups == subject
            subject_y = y[subject_mask]
            subject_predictions = predictions[subject_mask]
            subject_balanced_accuracy = (
                float(balanced_accuracy_score(subject_y, subject_predictions))
                if np.unique(subject_y).size == 2
                else float("nan")
            )
            subject_metrics.append(
                {
                    "subject_id": int(subject),
                    "windows": int(subject_mask.sum()),
                    "accuracy": float(
                        accuracy_score(subject_y, subject_predictions)
                    ),
                    "balanced_accuracy": subject_balanced_accuracy,
                    "f1_anxiety": float(
                        f1_score(
                            subject_y,
                            subject_predictions,
                            labels=[0, 1],
                            average="binary",
                            pos_label=1,
                            zero_division=0,
                        )
                    ),
                }
            )
        results[name] = {
            "pooled_accuracy": float(accuracy_score(y, predictions)),
            "pooled_balanced_accuracy": float(
                balanced_accuracy_score(y, predictions)
            ),
            "pooled_f1_anxiety": float(
                f1_score(y, predictions, zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(
                y, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y,
                predictions,
                labels=[0, 1],
                target_names=["No Anxiety", "Anxiety"],
                output_dict=True,
                zero_division=0,
            ),
            "macro_subject_accuracy": float(
                np.mean([item["accuracy"] for item in subject_metrics])
            ),
            "macro_subject_balanced_accuracy": float(
                np.nanmean(
                    [item["balanced_accuracy"] for item in subject_metrics]
                )
            ),
            "macro_subject_f1_anxiety": float(
                np.mean([item["f1_anxiety"] for item in subject_metrics])
            ),
            "per_subject": subject_metrics,
        }
    return results


def main() -> None:
    data, feature_columns = load_dataset()
    y = data[TARGET].astype(int).to_numpy()
    groups = data[GROUP].astype(int).to_numpy()
    indices = np.arange(len(data))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    reference_results = evaluate_split(
        data, feature_columns, train_indices, test_indices, "reference_random"
    )

    grouped_splitter = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    grouped_train, grouped_test = next(
        grouped_splitter.split(indices, y, groups=groups)
    )
    grouped_results = evaluate_split(
        data, feature_columns, grouped_train, grouped_test, "grouped_subject"
    )
    loso_results = evaluate_loso(data, feature_columns)

    report = {
        "data_directory": str(DATA_DIR),
        "rows": int(len(data)),
        "subjects": sorted(data[GROUP].astype(int).unique().tolist()),
        "features": feature_columns,
        "window_seconds": 60,
        "shift_seconds": 6,
        "label": "condition_stai_mean - baseline_stai_mean >= 0.2",
        "reference_results": reference_results,
        "grouped_results": grouped_results,
        "loso_results": loso_results,
    }
    report_path = MODEL_DIR / "evaluation_results.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
