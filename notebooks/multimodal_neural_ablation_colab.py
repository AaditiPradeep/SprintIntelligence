# %% [markdown]
# # Sprint Intelligence — Improved Neural Model and Ablation Study
#
# Three experiments are run on the same chronological split:
# 1. Structured early activity only
# 2. Structured activity + frozen pre-trained MiniLM text embeddings
# 3. Structured activity + MiniLM text + workflow-event sequence
#
# All models use focal loss and missing-label masking.

# %%
!pip -q install sentence-transformers

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v3_expanded.csv

# %%
import io
import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
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
df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, 3000)
df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)
print("Dataset shape:", df.shape)

# %% [markdown]
# ## Safe features, targets, and one shared chronological split

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

n = len(df)
train_end, validation_end = int(n * 0.70), int(n * 0.85)
train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()
print(f"Train: {len(train_df):,}; Validation: {len(validation_df):,}; Test: {len(test_df):,}")

# %% [markdown]
# ## Structured inputs and frozen pre-trained MiniLM text embeddings

# %%
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_FEATURES),
    ("source", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATEGORICAL_FEATURES),
])
x_struct_train = preprocessor.fit_transform(train_df[STRUCTURED_FEATURES]).astype("float32")
x_struct_valid = preprocessor.transform(validation_df[STRUCTURED_FEATURES]).astype("float32")
x_struct_test = preprocessor.transform(test_df[STRUCTURED_FEATURES]).astype("float32")

# MiniLM is used only as a frozen encoder. TensorFlow/Keras trains the neural
# heads; PyTorch is used internally by SentenceTransformers for encoding only.
encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=encoder_device)

def encode_text(texts):
    return text_encoder.encode(
        texts.tolist(), batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

x_text_train = encode_text(train_df["issue_text_at_cutoff"])
x_text_valid = encode_text(validation_df["issue_text_at_cutoff"])
x_text_test = encode_text(test_df["issue_text_at_cutoff"])
print("Structured input shape:", x_struct_train.shape)
print("MiniLM text embedding shape:", x_text_train.shape)

# %% [markdown]
# ## Workflow sequence vocabulary and masked targets

# %%
sequence_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=200,
    output_mode="int",
    output_sequence_length=80,
    standardize=None,
    split="whitespace",
)
sequence_vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(train_df["early_event_sequence"].values).batch(256)
)
SEQUENCE_VOCABULARY = sequence_vectorizer.get_vocabulary()
print("Workflow sequence vocabulary size:", len(SEQUENCE_VOCABULARY))

def make_labels_and_masks(split):
    labels, masks = {}, {}
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(split[target], errors="coerce")
        masks[target] = values.notna().to_numpy(dtype="float32")
        labels[target] = values.fillna(0).to_numpy(dtype="float32").reshape(-1, 1)
    return labels, masks

y_train, mask_train = make_labels_and_masks(train_df)
y_valid, mask_valid = make_labels_and_masks(validation_df)
y_test, mask_test = make_labels_and_masks(test_df)

# %% [markdown]
# ## Small focal-loss multi-task architecture

# %%
def fresh_sequence_vectorizer():
    layer = tf.keras.layers.TextVectorization(
        max_tokens=200, output_mode="int", output_sequence_length=80,
        standardize=None, split="whitespace",
    )
    layer.set_vocabulary(SEQUENCE_VOCABULARY)
    return layer


def build_model(use_text=False, use_sequence=False):
    tf.keras.backend.clear_session()
    inputs, branches = [], []

    structured_input = tf.keras.Input(shape=(x_struct_train.shape[1],), name="structured_input")
    structured = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(structured_input)
    structured = tf.keras.layers.BatchNormalization()(structured)
    structured = tf.keras.layers.Dropout(0.20)(structured)
    inputs.append(structured_input)
    branches.append(structured)

    if use_text:
        text_input = tf.keras.Input(shape=(x_text_train.shape[1],), name="text_embedding_input")
        text = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(text_input)
        text = tf.keras.layers.Dropout(0.25)(text)
        inputs.append(text_input)
        branches.append(text)

    if use_sequence:
        sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
        tokens = fresh_sequence_vectorizer()(sequence_input)
        sequence = tf.keras.layers.Embedding(len(SEQUENCE_VOCABULARY), 16, mask_zero=True)(tokens)
        sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(sequence)
        sequence = tf.keras.layers.Dropout(0.20)(sequence)
        inputs.append(sequence_input)
        branches.append(sequence)

    merged = branches[0] if len(branches) == 1 else tf.keras.layers.Concatenate()(branches)
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(0.30)(merged)

    outputs = {
        target: tf.keras.layers.Dense(1, activation="sigmoid", name=target)(merged)
        for target in TARGET_COLUMNS
    }
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    focal = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, alpha=0.75, gamma=2.0
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss={target: focal for target in TARGET_COLUMNS},
    )
    return model


def model_inputs(split, structured, text, use_text, use_sequence):
    inputs = {"structured_input": structured}
    if use_text:
        inputs["text_embedding_input"] = text
    if use_sequence:
        inputs["sequence_input"] = split["early_event_sequence"].values.reshape(-1, 1)
    return inputs


def best_threshold(y_true, probabilities):
    candidates = np.arange(0.10, 0.91, 0.05)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def train_and_evaluate(name, use_text, use_sequence):
    model = build_model(use_text, use_sequence)
    x_train = model_inputs(train_df, x_struct_train, x_text_train, use_text, use_sequence)
    x_valid = model_inputs(validation_df, x_struct_valid, x_text_valid, use_text, use_sequence)
    x_test = model_inputs(test_df, x_struct_test, x_text_test, use_text, use_sequence)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, min_lr=1e-5, verbose=0),
    ]
    history = model.fit(
        x_train, y_train, sample_weight=mask_train,
        validation_data=(x_valid, y_valid, mask_valid),
        epochs=15, batch_size=128, callbacks=callbacks, verbose=1,
    )

    valid_predictions = model.predict(x_valid, batch_size=256, verbose=0)
    test_predictions = model.predict(x_test, batch_size=256, verbose=0)
    rows, details = [], {}
    for target, task_name in TARGETS.items():
        valid_mask = mask_valid[target].astype(bool)
        test_mask = mask_test[target].astype(bool)
        y_valid_true = y_valid[target].reshape(-1)[valid_mask].astype(int)
        valid_probabilities = valid_predictions[target].reshape(-1)[valid_mask]
        threshold = best_threshold(y_valid_true, valid_probabilities)

        y_true = y_test[target].reshape(-1)[test_mask].astype(int)
        probabilities = test_predictions[target].reshape(-1)[test_mask]
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "Ablation": name,
            "Task": task_name,
            "Threshold": threshold,
            "Eligible test records": int(test_mask.sum()),
            "Positive test records": int(y_true.sum()),
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1-score": f1_score(y_true, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, probabilities),
            "PR-AUC": average_precision_score(y_true, probabilities),
        })
        details[target] = (y_true, probabilities, predictions, threshold)
    return model, history, pd.DataFrame(rows), details

# %% [markdown]
# ## Run the ablation study

# %%
EXPERIMENTS = [
    ("Structured neural network", False, False),
    ("Structured + MiniLM text", True, False),
    ("Structured + MiniLM text + workflow sequence", True, True),
]

all_results, experiment_outputs = [], {}
for name, use_text, use_sequence in EXPERIMENTS:
    print(f"\n{'=' * 90}\nRunning: {name}")
    model, history, result_table, details = train_and_evaluate(name, use_text, use_sequence)
    all_results.append(result_table)
    experiment_outputs[name] = {"model": model, "history": history, "details": details}

results_df = pd.concat(all_results, ignore_index=True)
display(results_df.sort_values(["Task", "F1-score"], ascending=[True, False]).style.format({
    "Threshold": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
}))

results_df.to_csv("neural_ablation_results_v3.csv", index=False)
files.download("neural_ablation_results_v3.csv")

# %% [markdown]
# ## Compare F1-score and PR-AUC across modalities

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(data=results_df, x="Task", y="F1-score", hue="Ablation", ax=axes[0])
axes[0].set_title("F1-score by Ablation")
axes[0].tick_params(axis="x", rotation=15)
axes[0].legend(fontsize=8)

sns.barplot(data=results_df, x="Task", y="PR-AUC", hue="Ablation", ax=axes[1])
axes[1].set_title("PR-AUC by Ablation")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Confusion matrices for the best-F1 ablation of each task

# %%
for task_name in results_df["Task"].unique():
    best_row = results_df[results_df["Task"] == task_name].sort_values("F1-score", ascending=False).iloc[0]
    best_name = best_row["Ablation"]
    target = next(key for key, value in TARGETS.items() if value == task_name)
    y_true, probabilities, predictions, threshold = experiment_outputs[best_name]["details"][target]

    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, cmap="Blues")
    plt.title(f"{task_name}\nBest: {best_name}\nThreshold={threshold:.2f}")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Save the best neural model for each task reference
#
# The full multi-modal model remains the proposed architecture. The ablation
# table determines whether text and workflow sequences actually improve it.

# %%
with open("neural_ablation_config_v3.json", "w", encoding="utf-8") as file:
    json.dump({
        "experiments": [item[0] for item in EXPERIMENTS],
        "structured_features": preprocessor.get_feature_names_out().tolist(),
        "sequence_vocabulary": SEQUENCE_VOCABULARY,
        "targets": TARGETS,
    }, file, indent=2)
files.download("neural_ablation_config_v3.json")
