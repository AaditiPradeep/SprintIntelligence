# %% [markdown]
# # Sprint Intelligence — Task-Adaptive Multi-Modal Neural Model (v4)
#
# Improvements over the first neural model:
# - safe cutoff snapshots: priority, issue type, status, assignee and project
# - pre-trained MiniLM text embeddings
# - task-adaptive modality fusion
# - focal loss with missing-label masking
# - exact F1 threshold selected from the validation precision–recall curve

# %%
!pip -q install sentence-transformers

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v4.csv

# %%
import io
import json
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
    ConfusionMatrixDisplay, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

FILE_NAME = "training_dataset_v4.csv"
df = pd.read_csv(io.BytesIO(uploaded[FILE_NAME]))
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, 3000)
df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)
print("Dataset shape:", df.shape)

# %% [markdown]
# ## v4 safe inputs and chronological split

# %%
NUMERIC_FEATURES = [
    "created_year", "created_month", "comments_available",
    "assignee_available_at_cutoff", "early_comment_count",
    "early_avg_comment_length", "early_history_count",
    "early_changelog_item_count", "early_status_change_count",
    "early_assignee_change_count", "early_priority_change_count",
    "early_description_change_count", "early_summary_change_count",
    "early_component_change_count", "early_label_change_count",
    "early_developer_activity_count",
]
CATEGORICAL_FEATURES = [
    "source_collection", "project_key", "priority_at_cutoff",
    "issue_type_at_cutoff", "status_at_cutoff",
]
STRUCTURED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
TARGET_COLUMNS = list(TARGETS)

required = STRUCTURED_FEATURES + TARGET_COLUMNS + ["issue_text_at_cutoff", "early_event_sequence"]
missing = sorted(set(required) - set(df.columns))
assert not missing, f"Missing v4 columns: {missing}"

n = len(df)
train_end, validation_end = int(n * 0.70), int(n * 0.85)
train_df = df.iloc[:train_end].copy()
validation_df = df.iloc[train_end:validation_end].copy()
test_df = df.iloc[validation_end:].copy()
print(f"Train: {len(train_df):,}; Validation: {len(validation_df):,}; Test: {len(test_df):,}")

# %% [markdown]
# ## Structured inputs, frozen MiniLM embeddings, and event vocabulary

# %%
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC_FEATURES),
    # Rare projects are grouped, avoiding a huge project-key feature space.
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=20,
            sparse_output=False
        )),
    ]), CATEGORICAL_FEATURES),
])
x_struct_train = preprocessor.fit_transform(train_df[STRUCTURED_FEATURES]).astype("float32")
x_struct_valid = preprocessor.transform(validation_df[STRUCTURED_FEATURES]).astype("float32")
x_struct_test = preprocessor.transform(test_df[STRUCTURED_FEATURES]).astype("float32")

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def embed(text):
    return encoder.encode(
        text.tolist(), batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

x_text_train = embed(train_df["issue_text_at_cutoff"])
x_text_valid = embed(validation_df["issue_text_at_cutoff"])
x_text_test = embed(test_df["issue_text_at_cutoff"])

sequence_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=200, output_mode="int", output_sequence_length=80,
    standardize=None, split="whitespace"
)
sequence_vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(train_df["early_event_sequence"].values).batch(256)
)
SEQUENCE_VOCABULARY = sequence_vectorizer.get_vocabulary()
print("Structured dimension:", x_struct_train.shape[1])
print("Text embedding dimension:", x_text_train.shape[1])
print("Event vocabulary:", len(SEQUENCE_VOCABULARY))

# %% [markdown]
# ## Labels: mask missing targets, but do not use aggressive manual class weights

# %%
def labels_and_masks(split):
    labels, masks = {}, {}
    for target in TARGET_COLUMNS:
        values = pd.to_numeric(split[target], errors="coerce")
        labels[target] = values.fillna(0).to_numpy(dtype="float32").reshape(-1, 1)
        masks[target] = values.notna().to_numpy(dtype="float32")
    return labels, masks

y_train, mask_train = labels_and_masks(train_df)
y_valid, mask_valid = labels_and_masks(validation_df)
y_test, mask_test = labels_and_masks(test_df)

# %% [markdown]
# ## Task-adaptive model
#
# Requirement volatility uses structured + text because workflow sequences did
# not improve that task in the ablation. Resolution and reopen use all three
# modalities because event order provided useful additional signal.

# %%
tf.keras.backend.clear_session()

structured_input = tf.keras.Input(shape=(x_struct_train.shape[1],), name="structured_input")
structured = tf.keras.layers.Dense(48, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(structured_input)
structured = tf.keras.layers.BatchNormalization()(structured)
structured = tf.keras.layers.Dropout(0.20)(structured)

text_input = tf.keras.Input(shape=(x_text_train.shape[1],), name="text_embedding_input")
text = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(text_input)
text = tf.keras.layers.Dropout(0.25)(text)

sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
sequence_layer = tf.keras.layers.TextVectorization(
    max_tokens=200, output_mode="int", output_sequence_length=80,
    standardize=None, split="whitespace"
)
sequence_layer.set_vocabulary(SEQUENCE_VOCABULARY)
sequence = sequence_layer(sequence_input)
sequence = tf.keras.layers.Embedding(len(SEQUENCE_VOCABULARY), 16, mask_zero=True)(sequence)
sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(sequence)
sequence = tf.keras.layers.Dropout(0.20)(sequence)

def task_head(inputs, name):
    x = tf.keras.layers.Concatenate()(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    return tf.keras.layers.Dense(1, activation="sigmoid", name=name)(x)

outputs = {
    "requirement_volatility_label": task_head([structured, text], "requirement_volatility_label"),
    "issue_resolution_risk_label": task_head([structured, text, sequence], "issue_resolution_risk_label"),
    "issue_reopen_label": task_head([structured, text, sequence], "issue_reopen_label"),
}

model = tf.keras.Model(
    inputs=[structured_input, text_input, sequence_input], outputs=outputs,
    name="task_adaptive_multimodal_sprint_intelligence"
)
focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True, alpha=0.75, gamma=2.0
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss={target: focal_loss for target in TARGET_COLUMNS},
)
model.summary()

# %% [markdown]
# ## Train with early stopping

# %%
def make_inputs(split, structured, text):
    return {
        "structured_input": structured,
        "text_embedding_input": text,
        "sequence_input": split["early_event_sequence"].values.reshape(-1, 1),
    }

x_train = make_inputs(train_df, x_struct_train, x_text_train)
x_valid = make_inputs(validation_df, x_struct_valid, x_text_valid)
x_test = make_inputs(test_df, x_struct_test, x_text_test)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1),
]
history = model.fit(
    x_train, y_train, sample_weight=mask_train,
    validation_data=(x_valid, y_valid, mask_valid),
    epochs=15, batch_size=128, callbacks=callbacks, verbose=1,
)

plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Task-Adaptive Neural Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown]
# ## Exact validation threshold selection and held-out test evaluation

# %%
def exact_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(np.clip(thresholds[np.nanargmax(f1_values)], 0.01, 0.99))

valid_predictions = model.predict(x_valid, batch_size=256, verbose=0)
test_predictions = model.predict(x_test, batch_size=256, verbose=0)

results, details = [], {}
for target, task_name in TARGETS.items():
    valid_available = mask_valid[target].astype(bool)
    test_available = mask_test[target].astype(bool)
    y_valid_true = y_valid[target].reshape(-1)[valid_available].astype(int)
    p_valid = valid_predictions[target].reshape(-1)[valid_available]
    threshold = exact_f1_threshold(y_valid_true, p_valid)

    y_true = y_test[target].reshape(-1)[test_available].astype(int)
    probabilities = test_predictions[target].reshape(-1)[test_available]
    predictions = (probabilities >= threshold).astype(int)
    results.append({
        "Task": task_name,
        "Model": "Task-Adaptive Multi-Modal Neural Network v4",
        "Threshold": threshold,
        "Eligible test records": int(test_available.sum()),
        "Positive test records": int(y_true.sum()),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
    })
    details[target] = (y_true, probabilities, predictions, threshold)

results_df = pd.DataFrame(results)
display(results_df.style.format({
    "Threshold": "{:.4f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
}))
results_df.to_csv("task_adaptive_neural_results_v4.csv", index=False)
files.download("task_adaptive_neural_results_v4.csv")

# %% [markdown]
# ## Test-set diagnostic charts

# %%
for target, task_name in TARGETS.items():
    y_true, probabilities, predictions, threshold = details[target]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{task_name}\nThreshold={threshold:.3f}")

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    axes[1].plot(recall, precision, color="#2563eb")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR-AUC={average_precision_score(y_true, probabilities):.3f}")
    plt.tight_layout()
    plt.show()

# %%
model.save("task_adaptive_multimodal_v4.keras")
with open("task_adaptive_multimodal_v4_artifacts.json", "w", encoding="utf-8") as file:
    json.dump({
        "structured_features": preprocessor.get_feature_names_out().tolist(),
        "sequence_vocabulary": SEQUENCE_VOCABULARY,
        "targets": TARGETS,
    }, file, indent=2)
files.download("task_adaptive_multimodal_v4_artifacts.json")
