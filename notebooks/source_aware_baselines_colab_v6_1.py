# Sprint Intelligence — v6.1 source-aware baseline comparison (Google Colab)
# Run this notebook before the neural notebook. It downloads the exact split
# file that the neural notebook reuses.

# %%
!pip -q install xgboost lightgbm

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v6_1.csv and model_feature_columns_v6.json

# %%
import io, json, random, warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE, TOP_K_FEATURES = 42, 30
np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE)
DATA_FILE, FEATURE_FILE = "training_dataset_v6_1.csv", "model_feature_columns_v6.json"
df = pd.read_csv(io.BytesIO(uploaded[DATA_FILE]), low_memory=False)
safe_features = json.loads(uploaded[FEATURE_FILE].decode("utf-8"))
# Public Jira exports contain more than one valid timestamp format.  Pandas'
# default single-format inference silently discarded JiraEcosystem records.
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["created_at", "source_collection", "issue_key"]).copy()
if len(df) < 39000:
    raise ValueError(
        f"Expected the 40,013-row v6.1 dataset, but loaded only {len(df):,} usable rows. "
        "Restart the Colab runtime, rerun cells from the top, and upload training_dataset_v6_1.csv."
    )

TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
CATEGORICAL = ["source_collection", "priority_at_cutoff", "issue_type_at_cutoff", "status_at_cutoff"]
NUMERIC = [column for column in safe_features if column not in CATEGORICAL]
INCLUDE_SOURCE_FEATURE = True  # Set False only for the source-dependence ablation.
if not INCLUDE_SOURCE_FEATURE:
    CATEGORICAL.remove("source_collection")
print(f"Rows: {len(df):,}; numeric={len(NUMERIC)}; categorical={CATEGORICAL}")

# %%
def source_aware_split(frame):
    """Chronological 70/15/15 split within every source collection."""
    parts = []
    source_sizes = frame.groupby("source_collection").size().to_dict()
    print("Eligible records by source:", source_sizes)
    for source, group in frame.groupby("source_collection", sort=True):
        group = group.sort_values(["created_at", "issue_key"], kind="stable").copy()
        n = len(group)
        if n < 3:
            raise ValueError(f"{source} has only {n} eligible rows. Loaded source counts: {source_sizes}")
        train_end = max(1, int(n * .70))
        valid_end = min(n - 1, max(train_end + 1, int(n * .85)))
        group["split"] = "test"
        group.iloc[:train_end, group.columns.get_loc("split")] = "train"
        group.iloc[train_end:valid_end, group.columns.get_loc("split")] = "validation"
        parts.append(group)
    return pd.concat(parts, ignore_index=True)

def choose_threshold(y, probability):
    p, r, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(np.clip(thresholds[int(np.nanargmax(f1))], .01, .99)) if len(thresholds) else .50

def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1-score": f1_score(y, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probability),
        "PR-AUC": average_precision_score(y, probability),
    }

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])

def model_set(y_train):
    ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.04, subsample=.85, colsample_bytree=.85, scale_pos_weight=ratio, eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(n_estimators=400, learning_rate=.04, num_leaves=31, subsample=.85, colsample_bytree=.85, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1),
    }

# %%
all_results, all_splits, all_selected_features = [], [], []
for target, task_name in TARGETS.items():
    task = df.dropna(subset=[target]).copy()
    task[target] = task[target].astype(int)
    task = source_aware_split(task)
    train, valid, test = (task.loc[task["split"] == name].copy() for name in ["train", "validation", "test"])
    y_train, y_valid, y_test = train[target].to_numpy(), valid[target].to_numpy(), test[target].to_numpy()
    if min(y_train.sum(), y_valid.sum(), y_test.sum()) == 0:
        raise ValueError(f"{task_name} has a partition without positive examples.")
    all_splits.append(task[["source_collection", "issue_key", "created_at", "split"]].assign(target=target))

    processor = make_preprocessor()
    x_train = processor.fit_transform(train[NUMERIC + CATEGORICAL])
    x_valid, x_test = processor.transform(valid[NUMERIC + CATEGORICAL]), processor.transform(test[NUMERIC + CATEGORICAL])
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, x_train.shape[1])).fit(x_train, y_train)
    x_train, x_valid, x_test = selector.transform(x_train), selector.transform(x_valid), selector.transform(x_test)
    names = processor.get_feature_names_out()[selector.get_support()]
    all_selected_features.extend({"Task": task_name, "Feature": name} for name in names)
    print(f"\n{task_name}: train/validation/test={len(train):,}/{len(valid):,}/{len(test):,}; positives={y_train.sum()}/{y_valid.sum()}/{y_test.sum()}")

    for model_name, model in model_set(y_train).items():
        model.fit(x_train, y_train)
        valid_probability = model.predict_proba(x_valid)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
        threshold = choose_threshold(y_valid, valid_probability)  # Never use test data here.
        result = metrics(y_test, test_probability, threshold)
        all_results.append({"Task": task_name, "Model": model_name, "Feature variant": "with source" if INCLUDE_SOURCE_FEATURE else "without source", "Selected feature count": x_train.shape[1], "Threshold (validation selected)": threshold, "Eligible test records": len(test), "Positive test records": int(y_test.sum()), **result})
        print(f"  {model_name}: F1={result['F1-score']:.3f}, PR-AUC={result['PR-AUC']:.3f}, ROC-AUC={result['ROC-AUC']:.3f}")

# %%
results = pd.DataFrame(all_results).sort_values(["Task", "PR-AUC"], ascending=[True, False])
selected_features = pd.DataFrame(all_selected_features)
splits = pd.concat(all_splits, ignore_index=True)
display(results.style.format({"Threshold (validation selected)": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"}))
display(selected_features.groupby("Task")["Feature"].apply(list).to_frame("Training-only selected features"))
results.to_csv("source_aware_baseline_results_v6_1_fixed.csv", index=False)
selected_features.to_csv("source_aware_selected_features_v6_1_fixed.csv", index=False)
splits.to_csv("source_aware_splits_v6_1_fixed.csv", index=False)
files.download("source_aware_baseline_results_v6_1_fixed.csv")
files.download("source_aware_selected_features_v6_1_fixed.csv")
files.download("source_aware_splits_v6_1_fixed.csv")
