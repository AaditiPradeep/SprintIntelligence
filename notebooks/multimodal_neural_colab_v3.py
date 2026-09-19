# %% [markdown]
# # Sprint Intelligence — Multi-Modal Multi-Task Neural Model
#
# Inputs (all available in the first 30 days):
# 1. structured Jira activity features
# 2. issue summary/description/early comments
# 3. ordered workflow-event sequence
#
# Outputs:
# - Requirement Volatility Risk
# - Issue Resolution Risk
# - Issue Reopen Risk

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v3_expanded.csv

# %%
import io
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

FILE_NAME = "training_dataset_v3_expanded.csv"
df = pd.read_csv(io.BytesIO(uploaded[FILE_NAME]))
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

# Cap text to keep GPU/RAM requirements manageable. Text was already restricted
# to the safe 30-day observation window during v3 creation.
df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, 3000)
df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)

print("Dataset shape:", df.shape)
display(df[["source_collection", "issue_text_at_cutoff", "early_event_sequence"]].head(3))

# %% [markdown]
# ## Safe structured inputs and chronological split

# %%
NUMERIC_FEATURES = [
    "created_year", "created_month", "comments_available",
    "early_comment_count", "early_avg_comment_length",
    "early_history_count", "early_changelog_item_count",
    "early_status_change_count", "early_assignee_change_count",
    "early_priority_change_count", "early_description_change_count",
    "early_summary_change_count", "early_component_change_count",
    "early_label_change_count", "early_developer_activity_count",
]
CATEGORICAL_FEATURES = ["source_collection"]
STRUCTURED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
TARGET_COLUMNS = list(TARGETS)

missing_columns = sorted(set(STRUCTURED_FEATURES + TARGET_COLUMNS) - set(df.columns))
assert not missing_columns, f"Missing expected columns: {missing_columns}"

# A single common timeline is used for all model inputs. Missing labels are
# masked in the loss; for example, unresolved issues do not train the delay head.
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)
train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()

print(f"Train: {len(train_df):,} | Validation: {len(validation_df):,} | Test: {len(test_df):,}")
for target, name in TARGETS.items():
    for split_name, split in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
        eligible = split[target].notna()
        rate = split.loc[eligible, target].mean() if eligible.any() else np.nan
        print(f"{name:30} {split_name:10} eligible={eligible.sum():5d}, positive rate={rate:.2%}")

# %% [markdown]
# ## Prepare structured, text, and sequence inputs

# %%
structured_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("source", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])

x_structured_train = structured_preprocessor.fit_transform(train_df[STRUCTURED_FEATURES]).astype("float32")
x_structured_validation = structured_preprocessor.transform(validation_df[STRUCTURED_FEATURES]).astype("float32")
x_structured_test = structured_preprocessor.transform(test_df[STRUCTURED_FEATURES]).astype("float32")

# Fit vocabulary exclusively on training text/event sequences.
text_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=200,
    standardize="lower_and_strip_punctuation",
)
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_df["issue_text_at_cutoff"].values).batch(256))

sequence_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=200,
    output_mode="int",
    output_sequence_length=80,
    standardize=None,
    split="whitespace",
)
sequence_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_df["early_event_sequence"].values).batch(256))

def prepare_inputs(split, structured):
    return {
        "structured_input": structured,
        "text_input": split["issue_text_at_cutoff"].values.reshape(-1, 1),
        "sequence_input": split["early_event_sequence"].values.reshape(-1, 1),
    }

x_train = prepare_inputs(train_df, x_structured_train)
x_validation = prepare_inputs(validation_df, x_structured_validation)
x_test = prepare_inputs(test_df, x_structured_test)

print("Structured feature dimension:", x_structured_train.shape[1])
print("Text vocabulary size:", len(text_vectorizer.get_vocabulary()))
print("Workflow vocabulary size:", len(sequence_vectorizer.get_vocabulary()))

# %% [markdown]
# ## Build masked targets and class weights

# %%
def masked_targets_and_weights(split, reference_train):
    labels, weights = {}, {}
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(split[target], errors="coerce")
        mask = values.notna().to_numpy().astype("float32")
        # Missing targets are set to zero but receive zero sample weight.
        labels[target] = values.fillna(0).to_numpy(dtype="float32").reshape(-1, 1)

        train_values = pd.to_numeric(reference_train[target], errors="coerce").dropna()
        positives = max(float((train_values == 1).sum()), 1.0)
        negatives = max(float((train_values == 0).sum()), 1.0)
        positive_weight = negatives / positives
        weights[target] = mask * np.where(labels[target].reshape(-1) == 1, positive_weight, 1.0)
        print(f"{TARGETS[target]}: positive class weight = {positive_weight:.2f}")
    return labels, weights

y_train, weights_train = masked_targets_and_weights(train_df, train_df)
y_validation, weights_validation = masked_targets_and_weights(validation_df, train_df)
y_test, _ = masked_targets_and_weights(test_df, train_df)

# %% [markdown]
# ## Multi-modal, multi-task neural architecture

# %%
tf.keras.backend.clear_session()

structured_input = tf.keras.Input(shape=(x_structured_train.shape[1],), name="structured_input")
structured_branch = tf.keras.layers.Dense(64, activation="relu")(structured_input)
structured_branch = tf.keras.layers.BatchNormalization()(structured_branch)
structured_branch = tf.keras.layers.Dropout(0.25)(structured_branch)

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
text_tokens = text_vectorizer(text_input)
text_branch = tf.keras.layers.Embedding(len(text_vectorizer.get_vocabulary()), 96, mask_zero=True)(text_tokens)
text_branch = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(48))(text_branch)
text_branch = tf.keras.layers.Dropout(0.25)(text_branch)

sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
sequence_tokens = sequence_vectorizer(sequence_input)
sequence_branch = tf.keras.layers.Embedding(len(sequence_vectorizer.get_vocabulary()), 32, mask_zero=True)(sequence_tokens)
sequence_branch = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24))(sequence_branch)

merged = tf.keras.layers.Concatenate()([structured_branch, text_branch, sequence_branch])
merged = tf.keras.layers.Dense(128, activation="relu")(merged)
merged = tf.keras.layers.BatchNormalization()(merged)
merged = tf.keras.layers.Dropout(0.35)(merged)
merged = tf.keras.layers.Dense(64, activation="relu")(merged)

outputs = {
    target: tf.keras.layers.Dense(1, activation="sigmoid", name=target)(merged)
    for target in TARGET_COLUMNS
}

model = tf.keras.Model(
    inputs=[structured_input, text_input, sequence_input],
    outputs=outputs,
    name="sprint_intelligence_multimodal_multitask",
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={target: tf.keras.losses.BinaryCrossentropy() for target in TARGET_COLUMNS},
)
model.summary()

# %% [markdown]
# ## Train

# %%
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1
    ),
]

history = model.fit(
    x_train,
    y_train,
    sample_weight=weights_train,
    validation_data=(x_validation, y_validation, weights_validation),
    epochs=20,
    batch_size=128,
    callbacks=callbacks,
    verbose=1,
)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Multi-Task Neural Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Tune thresholds on validation and evaluate once on the held-out test set

# %%
def best_f1_threshold(y_true, probabilities):
    candidates = np.arange(0.10, 0.91, 0.05)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])

validation_predictions = model.predict(x_validation, batch_size=256, verbose=0)
test_predictions = model.predict(x_test, batch_size=256, verbose=0)

results = []
prediction_details = {}
for target, task_name in TARGETS.items():
    validation_mask = validation_df[target].notna().to_numpy()
    test_mask = test_df[target].notna().to_numpy()

    y_valid_true = validation_df.loc[validation_mask, target].astype(int).to_numpy()
    valid_probabilities = validation_predictions[target].reshape(-1)[validation_mask]
    threshold = best_f1_threshold(y_valid_true, valid_probabilities)

    y_true = test_df.loc[test_mask, target].astype(int).to_numpy()
    probabilities = test_predictions[target].reshape(-1)[test_mask]
    predictions = (probabilities >= threshold).astype(int)

    result = {
        "Task": task_name,
        "Model": "Multi-Modal Multi-Task Neural Network",
        "Threshold": threshold,
        "Eligible test records": int(test_mask.sum()),
        "Positive test records": int(y_true.sum()),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    }
    results.append(result)
    prediction_details[target] = (y_true, probabilities, predictions, threshold)

results_df = pd.DataFrame(results)
display(results_df.style.format({
    "Threshold": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
}))

results_df.to_csv("multimodal_neural_model_results_v3.csv", index=False)
files.download("multimodal_neural_model_results_v3.csv")

# %% [markdown]
# ## Confusion matrices, ROC curves, and Precision–Recall curves

# %%
for target, task_name in TARGETS.items():
    y_true, probabilities, predictions, threshold = prediction_details[target]
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{task_name}\nThreshold = {threshold:.2f}")

    axes[1].plot(fpr, tpr, color="#1d4ed8")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.6)
    axes[1].set_title(f"ROC Curve — AUC={roc_auc_score(y_true, probabilities):.3f}")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")

    axes[2].plot(recall, precision, color="#dc2626")
    axes[2].set_title(f"Precision–Recall — AUC={average_precision_score(y_true, probabilities):.3f}")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    plt.suptitle(task_name, y=1.05, fontsize=14)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Save the proposed model and vectorizer vocabularies

# %%
model.save("sprint_intelligence_multimodal_multitask.keras")

artifacts = {
    "structured_feature_names": structured_preprocessor.get_feature_names_out().tolist(),
    "text_vocabulary": text_vectorizer.get_vocabulary(),
    "sequence_vocabulary": sequence_vectorizer.get_vocabulary(),
    "targets": TARGETS,
}
import json
with open("multimodal_model_artifacts.json", "w", encoding="utf-8") as file:
    json.dump(artifacts, file, indent=2)

files.download("sprint_intelligence_multimodal_multitask.keras")
files.download("multimodal_model_artifacts.json")
