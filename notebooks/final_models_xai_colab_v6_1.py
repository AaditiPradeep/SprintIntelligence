# Sprint Intelligence — final selected models and explainability (Google Colab)
# Uses the corrected v6.1 source-aware split. Upload these three files:
# training_dataset_v6_1.csv, model_feature_columns_v6.json,
# source_aware_splits_v6_1_fixed.csv.

# %%
!pip -q install shap joblib

# %%
from google.colab import files
uploaded = files.upload()

# %%
import io, json, random, warnings
import joblib
import numpy as np
import pandas as pd
import shap
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

DATA_FILE = "training_dataset_v6_1.csv"
FEATURE_FILE = "model_feature_columns_v6.json"
SPLIT_FILE = "source_aware_splits_v6_1_fixed.csv"
df = pd.read_csv(io.BytesIO(uploaded[DATA_FILE]), low_memory=False)
safe_features = json.loads(uploaded[FEATURE_FILE].decode("utf-8"))
splits = pd.read_csv(io.BytesIO(uploaded[SPLIT_FILE]))
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["created_at", "source_collection", "issue_key"]).copy()

CATEGORICAL = ["source_collection", "priority_at_cutoff", "issue_type_at_cutoff", "status_at_cutoff"]
NUMERIC = [column for column in safe_features if column not in CATEGORICAL]
SELECTED_MODELS = {
    "requirement_volatility_label": ("Requirement Volatility Risk", "Logistic Regression"),
    "issue_resolution_risk_label": ("Issue Resolution Risk", "Random Forest"),
    "issue_reopen_label": ("Issue Reopen Risk", "Random Forest"),
}

# %%
def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
        ("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])

def choose_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(np.clip(thresholds[np.nanargmax(f1)], .01, .99)) if len(thresholds) else .50

def score(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1-score": f1_score(y, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y, probability),
        "PR-AUC": average_precision_score(y, probability),
    }, predicted

def get_partitions(target):
    assignments = splits.loc[splits["target"] == target, ["source_collection", "issue_key", "split"]]
    task = df.dropna(subset=[target]).copy()
    task[target] = task[target].astype(int)
    task = task.merge(assignments, on=["source_collection", "issue_key"], how="inner", validate="one_to_one")
    if len(task) != len(assignments):
        raise ValueError(f"Split file does not match dataset for {target}.")
    return (task.loc[task["split"] == name].copy() for name in ["train", "validation", "test"])

def make_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)

# %%
results, explanation_rows, prediction_rows = [], [], []
for target, (task_name, model_name) in SELECTED_MODELS.items():
    train, valid, test = get_partitions(target)
    y_train, y_valid, y_test = train[target].to_numpy(), valid[target].to_numpy(), test[target].to_numpy()
    processor = make_preprocessor()
    x_train = processor.fit_transform(train[NUMERIC + CATEGORICAL])
    x_valid = processor.transform(valid[NUMERIC + CATEGORICAL])
    x_test = processor.transform(test[NUMERIC + CATEGORICAL])
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, x_train.shape[1])).fit(x_train, y_train)
    x_train, x_valid, x_test = selector.transform(x_train), selector.transform(x_valid), selector.transform(x_test)
    selected_names = processor.get_feature_names_out()[selector.get_support()]

    model = make_model(model_name)
    model.fit(x_train, y_train)
    valid_probability = model.predict_proba(x_valid)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_valid, valid_probability)
    metrics, prediction = score(y_test, test_probability, threshold)
    results.append({"Task": task_name, "Selected model": model_name, "Threshold (validation selected)": threshold, "Eligible test records": len(test), "Positive test records": int(y_test.sum()), **metrics})
    prediction_rows.append(pd.DataFrame({"Task": task_name, "source_collection": test["source_collection"], "issue_key": test["issue_key"], "True label": y_test, "Risk probability": test_probability, "Risk prediction": prediction}))

    if model_name == "Logistic Regression":
        importance = np.abs(model.coef_.ravel())
        signed = model.coef_.ravel()
    else:
        importance = model.feature_importances_
        signed = model.feature_importances_
    explanation_rows.extend({"Task": task_name, "Feature": feature, "Importance": float(value), "Signed coefficient / importance": float(direction)} for feature, value, direction in zip(selected_names, importance, signed))

    artifact = {"processor": processor, "selector": selector, "model": model, "selected_feature_names": selected_names.tolist(), "threshold": threshold, "numeric_features": NUMERIC, "categorical_features": CATEGORICAL}
    safe_name = target.replace("_label", "")
    joblib.dump(artifact, f"{safe_name}_selected_model_v6_1.joblib")
    files.download(f"{safe_name}_selected_model_v6_1.joblib")

# %%
final_results = pd.DataFrame(results)
feature_importance = pd.DataFrame(explanation_rows).sort_values(["Task", "Importance"], ascending=[True, False])
test_predictions = pd.concat(prediction_rows, ignore_index=True)
display(final_results.style.format({"Threshold (validation selected)": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}"}))
display(feature_importance.groupby("Task").head(10))
final_results.to_csv("final_selected_model_results_v6_1.csv", index=False)
feature_importance.to_csv("global_feature_importance_v6_1.csv", index=False)
test_predictions.to_csv("test_risk_predictions_v6_1.csv", index=False)
files.download("final_selected_model_results_v6_1.csv")
files.download("global_feature_importance_v6_1.csv")
files.download("test_risk_predictions_v6_1.csv")
