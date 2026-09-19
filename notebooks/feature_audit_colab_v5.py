# %% [markdown]
# # Sprint Intelligence — Timing and Velocity Feature Audit (v5)
#
# This notebook audits only features known within the 30-day observation window.
# It measures missingness, low/high-risk separation, mutual information, and
# chronological train/test drift before the features enter neural models.

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v5.csv

# %%
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

sns.set_theme(style="whitegrid")
FILE_NAME = "training_dataset_v5.csv"
df = pd.read_csv(io.BytesIO(uploaded[FILE_NAME]))
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

TIMING_FEATURES = [
    "days_to_first_comment",
    "days_to_first_assignee_change",
    "days_to_first_status_change",
    "days_to_first_activity",
    "max_inactivity_gap_days",
    "early_active_days",
    "early_events_per_day",
    "early_comments_per_day",
    "unique_early_assignees",
    "unique_early_commenters",
    "unique_early_changelog_authors",
]
TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
missing = sorted(set(TIMING_FEATURES) - set(df.columns))
assert not missing, f"Missing timing features: {missing}"

n = len(df)
train_df = df.iloc[:int(n * 0.70)].copy()
test_df = df.iloc[int(n * 0.85):].copy()
print(f"Rows — train: {len(train_df):,}; test: {len(test_df):,}")

# %% [markdown]
# ## Audit table: completeness, class separation, drift, and mutual information

# %%
audit_rows = []
for target, task_name in TARGETS.items():
    train_task = train_df.dropna(subset=[target]).copy()
    test_task = test_df.dropna(subset=[target]).copy()
    y_train = train_task[target].astype(int)

    x_imputed = SimpleImputer(strategy="median").fit_transform(train_task[TIMING_FEATURES])
    mi_scores = mutual_info_classif(x_imputed, y_train, random_state=42)

    for feature, mi in zip(TIMING_FEATURES, mi_scores):
        low = train_task.loc[y_train == 0, feature]
        high = train_task.loc[y_train == 1, feature]
        train_median = train_task[feature].median()
        test_median = test_task[feature].median()
        median_drift = abs(test_median - train_median) / (abs(train_median) + 1e-6)
        audit_rows.append({
            "Task": task_name,
            "Feature": feature,
            "Train missing %": train_task[feature].isna().mean() * 100,
            "Test missing %": test_task[feature].isna().mean() * 100,
            "Train zero %": (train_task[feature] == 0).mean() * 100,
            "Low-risk median": low.median(),
            "High-risk median": high.median(),
            "Train median": train_median,
            "Test median": test_median,
            "Relative median drift": median_drift,
            "Mutual Information": mi,
        })

audit_df = pd.DataFrame(audit_rows)
audit_df["Recommendation"] = np.where(
    (audit_df["Mutual Information"] >= 0.005)
    & (audit_df["Train missing %"] <= 20)
    & (audit_df["Relative median drift"] <= 1.0),
    "Candidate for model", "Review or exclude"
)

for task_name in audit_df["Task"].unique():
    print(f"\n{task_name}")
    display(
        audit_df[audit_df["Task"] == task_name]
        .sort_values("Mutual Information", ascending=False)
        .style.format({
            "Train missing %": "{:.1f}%", "Test missing %": "{:.1f}%",
            "Train zero %": "{:.1f}%", "Relative median drift": "{:.2f}",
            "Mutual Information": "{:.4f}",
        })
    )

audit_df.to_csv("timing_feature_audit_v5.csv", index=False)
files.download("timing_feature_audit_v5.csv")

# %% [markdown]
# ## Visual checks: top timing features for each risk task

# %%
for target, task_name in TARGETS.items():
    top_features = (
        audit_df[audit_df["Task"] == task_name]
        .sort_values("Mutual Information", ascending=False)
        .head(4)["Feature"].tolist()
    )
    task_data = train_df.dropna(subset=[target]).copy()
    task_data[target] = task_data[target].astype(int)
    fig, axes = plt.subplots(1, len(top_features), figsize=(5 * len(top_features), 4))
    if len(top_features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, top_features):
        upper = task_data[feature].quantile(0.99)
        plot_data = task_data[task_data[feature] <= upper]
        sns.boxplot(data=plot_data, x=target, y=feature, hue=target,
                    palette=["#74c69d", "#e63946"], legend=False, ax=ax)
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("Risk label")
    plt.suptitle(f"Top Timing Features — {task_name}", y=1.04)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Export the approved timing features per task

# %%
approved = {
    task_name: (
        audit_df[(audit_df["Task"] == task_name)
                 & (audit_df["Recommendation"] == "Candidate for model")]
        .sort_values("Mutual Information", ascending=False)
        .head(6)["Feature"].tolist()
    )
    for task_name in TARGETS.values()
}
print(json.dumps(approved, indent=2))
with open("approved_timing_features_v5.json", "w", encoding="utf-8") as file:
    json.dump(approved, file, indent=2)
files.download("approved_timing_features_v5.json")
