# %% [markdown]
# # Sprint Intelligence — Baseline Model Training (v3)
#
# This notebook trains Logistic Regression, Random Forest, XGBoost, and
# LightGBM separately for Requirement Volatility, Resolution Risk, and Reopen
# Risk. It uses only leakage-safe structured features from the first 30 days.

# %%
# Run this cell once in Google Colab.
!pip -q install xgboost lightgbm

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v3_expanded.csv

# %%
import io
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
FILE_NAME = "training_dataset_v3_expanded.csv"

df = pd.read_csv(io.BytesIO(uploaded[FILE_NAME]))
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

print("Dataset shape:", df.shape)
display(df.head(3))

# %% [markdown]
# ## Safe inputs and targets
#
# Do not add issue key, future-event fields, resolution duration, final status,
# text fields, or labels to this baseline. Text and sequences are reserved for
# the later neural model.

# %%
NUMERIC_FEATURES = [
    "created_year",
    "created_month",
    "comments_available",
    "early_comment_count",
    "early_avg_comment_length",
    "early_history_count",
    "early_changelog_item_count",
    "early_status_change_count",
    "early_assignee_change_count",
    "early_priority_change_count",
    "early_description_change_count",
    "early_summary_change_count",
    "early_component_change_count",
    "early_label_change_count",
    "early_developer_activity_count",
]
CATEGORICAL_FEATURES = ["source_collection"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
N_SELECTED_FEATURES = 15

TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}

missing_features = sorted(set(FEATURES) - set(df.columns))
assert not missing_features, f"Missing expected features: {missing_features}"
print("Safe baseline features:", len(FEATURES))

# %%
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_FEATURES),
        ]
    )


def make_model_pipeline(estimator):
    """Feature selection is fitted inside the pipeline, using training data only."""
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=N_SELECTED_FEATURES)),
        ("model", estimator),
    ])


def chronological_split(data, target):
    """70% train, 15% validation, 15% test, ordered by issue creation date."""
    task_df = data.dropna(subset=[target]).copy().sort_values("created_at")
    task_df[target] = task_df[target].astype(int)
    n = len(task_df)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    train = task_df.iloc[:train_end]
    validation = task_df.iloc[train_end:validation_end]
    test = task_df.iloc[validation_end:]
    return train, validation, test


def build_models(y_train):
    positives = max(int(y_train.sum()), 1)
    negatives = max(int((y_train == 0).sum()), 1)
    positive_weight = negatives / positives

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            scale_pos_weight=positive_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85,
            class_weight={0: 1, 1: positive_weight},
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1
        ),
    }


def evaluate(name, fitted_pipeline, x_data, y_true, split_name, threshold=0.50):
    probabilities = fitted_pipeline.predict_proba(x_data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Model": name,
        "Split": split_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "y_true": y_true.to_numpy(),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def choose_threshold(y_true, probabilities):
    """Select threshold using validation F1; avoids choosing it on test data."""
    candidates = np.arange(0.20, 0.81, 0.05)
    scores = [f1_score(y_true, probabilities >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])

# %% [markdown]
# ## Train and compare all baseline models

# %%
all_results = []
trained_models = {}

for target, task_name in TARGETS.items():
    train_df, validation_df, test_df = chronological_split(df, target)
    x_train, y_train = train_df[FEATURES], train_df[target]
    x_validation, y_validation = validation_df[FEATURES], validation_df[target]
    x_test, y_test = test_df[FEATURES], test_df[target]

    print(f"\n{'=' * 80}\n{task_name}")
    print(f"Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}")
    print(f"Training positive-risk rate: {y_train.mean():.2%}")

    task_models = {}
    validation_rows = []
    for model_name, estimator in build_models(y_train).items():
        pipeline = make_model_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        validation_result = evaluate(model_name, pipeline, x_validation, y_validation, "Validation")
        validation_rows.append(validation_result)
        task_models[model_name] = pipeline

    validation_table = pd.DataFrame(validation_rows).drop(columns=["y_true", "probabilities", "predictions"])
    display(validation_table.sort_values("F1-score", ascending=False).style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
    }))

    # Tune threshold on validation only, then report final test performance.
    for model_name, pipeline in task_models.items():
        valid_probabilities = pipeline.predict_proba(x_validation)[:, 1]
        threshold = choose_threshold(y_validation, valid_probabilities)
        result = evaluate(model_name, pipeline, x_test, y_test, "Test", threshold)
        result["Task"] = task_name
        all_results.append(result)

    trained_models[target] = {
        "models": task_models,
        "test_df": test_df,
        "y_test": y_test,
        "task_name": task_name,
    }

# %%
metric_columns = ["Task", "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]
results_table = pd.DataFrame(all_results)[metric_columns].sort_values(["Task", "F1-score"], ascending=[True, False])
display(results_table.style.format({
    "Threshold": "{:.2f}", "Accuracy": "{:.3f}", "Precision": "{:.3f}",
    "Recall": "{:.3f}", "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
}))

results_table.to_csv("baseline_model_comparison_v3.csv", index=False)
files.download("baseline_model_comparison_v3.csv")

# %% [markdown]
# ## Feature-selection results
#
# Mutual information ranks features by how informative they are for each risk.
# This ranking is learned using the training split only. The selected 15
# features are then passed to every baseline model for that task.

# %%
for target, task_name in TARGETS.items():
    task_results = [row for row in all_results if row["Task"] == task_name]
    representative = max(task_results, key=lambda row: row["F1-score"])
    pipeline = trained_models[target]["models"][representative["Model"]]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selector = pipeline.named_steps["feature_selection"]
    selection_table = pd.DataFrame({
        "Feature": feature_names,
        "Mutual Information Score": selector.scores_,
        "Selected": selector.get_support(),
    }).sort_values("Mutual Information Score", ascending=False)

    print(f"\nFeature selection for: {task_name}")
    display(selection_table.style.format({"Mutual Information Score": "{:.4f}"}))
    selection_table.to_csv(
        f"selected_features_{target.replace('_label', '')}.csv", index=False
    )

# %% [markdown]
# ## Confusion matrices and ROC curves for the best baseline per task

# %%
for target, task_name in TARGETS.items():
    task_results = [row for row in all_results if row["Task"] == task_name]
    best = max(task_results, key=lambda row: row["F1-score"])

    print(f"\nBest baseline for {task_name}: {best['Model']} (F1 = {best['F1-score']:.3f})")
    print(classification_report(best["y_true"], best["predictions"], digits=3, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ConfusionMatrixDisplay.from_predictions(
        best["y_true"], best["predictions"], cmap="Blues", ax=axes[0]
    )
    axes[0].set_title(f"{task_name}\n{best['Model']} — Confusion Matrix")

    for row in task_results:
        fpr, tpr, _ = roc_curve(row["y_true"], row["probabilities"])
        axes[1].plot(fpr, tpr, label=f"{row['Model']} (AUC={row['ROC-AUC']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[1].set_title(f"{task_name} — ROC Curves")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Optional: feature importance for the best tree-based model

# %%
for target, task_name in TARGETS.items():
    candidate_rows = [row for row in all_results if row["Task"] == task_name and row["Model"] in {"XGBoost", "LightGBM", "Random Forest"}]
    best_tree = max(candidate_rows, key=lambda row: row["F1-score"])
    pipeline = trained_models[target]["models"][best_tree["Model"]]
    model = pipeline.named_steps["model"]
    transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    selected_mask = pipeline.named_steps["feature_selection"].get_support()
    selected_names = transformed_names[selected_mask]

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": selected_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)

        plt.figure(figsize=(9, 6))
        sns.barplot(data=importance, x="Importance", y="Feature", hue="Feature", legend=False, palette="crest")
        plt.title(f"Top Feature Importances — {task_name} ({best_tree['Model']})")
        plt.tight_layout()
        plt.show()
        display(importance)
