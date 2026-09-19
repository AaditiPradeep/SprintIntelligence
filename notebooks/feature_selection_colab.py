# %% [markdown]
# # Sprint Intelligence - Feature Selection
#
# Upload `jira_features_sample.csv` to Colab, then run the cells below.

# %%
from google.colab import files
uploaded = files.upload()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

csv_name = next(iter(uploaded.keys()))
df = pd.read_csv(csv_name)
df.head()

# %%
print("Shape:", df.shape)
print(df[[
    "requirement_volatility_label",
    "issue_resolution_risk_label",
    "issue_reopen_label",
]].sum())

# %% [markdown]
# ## Feature Groups
#
# Some columns are labels or direct outcome fields. They should not be used as model inputs.

# %%
targets = [
    "requirement_volatility_label",
    "issue_resolution_risk_label",
    "issue_reopen_label",
]

id_columns = ["issue_key"]
direct_outcome_columns = [
    "resolution_days",
    "is_resolved",
    "reopen_transition_count",
]

leakage_by_target = {
    "requirement_volatility_label": ["requirement_change_count"],
    "issue_resolution_risk_label": [],
    "issue_reopen_label": ["reopen_transition_count"],
}

categorical_columns = [
    "source_collection",
    "project_key",
    "issue_type",
    "priority",
    "status_category",
]

numeric_columns = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in targets + direct_outcome_columns
]

print("Numeric features:", numeric_columns)
print("Categorical features:", categorical_columns)

# %% [markdown]
# ## Correlation Check

# %%
corr = df[numeric_columns + targets].corr(numeric_only=True)
for target in targets:
    print("\nTop correlations for", target)
    print(corr[target].drop(target).abs().sort_values(ascending=False).head(10))

# %% [markdown]
# ## Mutual Information and Random Forest Importance

# %%
def build_feature_matrix(target):
    blocked = set(targets + id_columns + direct_outcome_columns + leakage_by_target[target])
    feature_cols = [col for col in numeric_columns + categorical_columns if col not in blocked]

    X = df[feature_cols].copy()
    y = df[target].astype(int)

    num_cols = [col for col in feature_cols if col in numeric_columns]
    cat_cols = [col for col in feature_cols if col in categorical_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    X_encoded = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return X_encoded, y, feature_names


def rank_features(target):
    X_encoded, y, feature_names = build_feature_matrix(target)

    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)
    mi_rank = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_encoded, y)
    rf_rank = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf.feature_importances_,
    }).sort_values("rf_importance", ascending=False)

    return mi_rank, rf_rank


rankings = {}
for target in targets:
    mi_rank, rf_rank = rank_features(target)
    rankings[target] = {"mi": mi_rank, "rf": rf_rank}
    print("\nTarget:", target)
    print("\nMutual information:")
    print(mi_rank.head(10).to_string(index=False))
    print("\nRandom Forest importance:")
    print(rf_rank.head(10).to_string(index=False))

# %% [markdown]
# ## Visualize Top Features

# %%
for target in targets:
    top = rankings[target]["rf"].head(10).sort_values("rf_importance")
    plt.figure(figsize=(8, 4))
    plt.barh(top["feature"], top["rf_importance"])
    plt.title(f"Top Random Forest Features - {target}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Save Feature Ranking Tables

# %%
for target in targets:
    rankings[target]["mi"].to_csv(f"{target}_mutual_information.csv", index=False)
    rankings[target]["rf"].to_csv(f"{target}_random_forest_importance.csv", index=False)

files.download("requirement_volatility_label_random_forest_importance.csv")
files.download("issue_resolution_risk_label_random_forest_importance.csv")
files.download("issue_reopen_label_random_forest_importance.csv")
