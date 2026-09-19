# %% [markdown]
# # Sprint Intelligence — Task-Specific Neural Models (v5)
#
# Run `feature_audit_colab_v5.py` first. Upload both the v5 dataset and its
# `approved_timing_features_v5.json` output here.

# %%
!pip -q install sentence-transformers

# %%
from google.colab import files
uploaded = files.upload()  # Upload training_dataset_v5.csv and approved_timing_features_v5.json

# %%
import io
import json
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)

FILE_NAME = "training_dataset_v5.csv"
APPROVED_NAME = "approved_timing_features_v5.json"
df = pd.read_csv(io.BytesIO(uploaded[FILE_NAME]))
with open(APPROVED_NAME, "r", encoding="utf-8") as file:
    approved_timing = json.load(file)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
df = df.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, 3000)
df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)

TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
BASE_NUMERIC = [
    "created_year", "created_month", "comments_available",
    "early_comment_count", "early_avg_comment_length",
    "early_history_count", "early_changelog_item_count",
    "early_status_change_count", "early_assignee_change_count",
    "early_priority_change_count", "early_description_change_count",
    "early_summary_change_count", "early_component_change_count",
    "early_label_change_count", "early_developer_activity_count",
]
CATEGORICAL = ["source_collection"]

# Set to True only after a successful fast run. Full tuning trains more models.
RUN_FULL_TUNING = False
CONFIGS = {
    "requirement_volatility_label": [
        {"name": "BCE, positive weight 1.0", "loss": "bce", "positive_scale": 1.0},
        {"name": "BCE, positive weight 1.5", "loss": "bce", "positive_scale": 1.5},
    ],
    "issue_resolution_risk_label": [
        {"name": "BCE, positive weight 1.0", "loss": "bce", "positive_scale": 1.0},
        {"name": "BCE, positive weight 1.5", "loss": "bce", "positive_scale": 1.5},
    ],
    "issue_reopen_label": [
        {"name": "Focal alpha=0.50 gamma=1", "loss": "focal", "alpha": 0.50, "gamma": 1.0},
        {"name": "Focal alpha=0.50 gamma=2", "loss": "focal", "alpha": 0.50, "gamma": 2.0},
        {"name": "Focal alpha=0.75 gamma=1", "loss": "focal", "alpha": 0.75, "gamma": 1.0},
        {"name": "Focal alpha=0.75 gamma=2", "loss": "focal", "alpha": 0.75, "gamma": 2.0},
    ],
}

print("Approved timing features:")
print(json.dumps(approved_timing, indent=2))

# %% [markdown]
# ## Shared frozen text encoder and workflow vocabulary

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def exact_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(np.clip(thresholds[np.nanargmax(f1_values)], 0.01, 0.99))


def sequence_vocabulary_for_train(train):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=200, output_mode="int", output_sequence_length=80,
        standardize=None, split="whitespace"
    )
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train["early_event_sequence"].values).batch(256))
    return vectorizer.get_vocabulary()


def fresh_sequence_layer(vocabulary):
    layer = tf.keras.layers.TextVectorization(
        max_tokens=200, output_mode="int", output_sequence_length=80,
        standardize=None, split="whitespace"
    )
    layer.set_vocabulary(vocabulary)
    return layer


def build_model(structured_dim, text_dim, use_sequence, config, sequence_vocabulary=None):
    tf.keras.backend.clear_session()
    structured_input = tf.keras.Input(shape=(structured_dim,), name="structured_input")
    structured = tf.keras.layers.Dense(48, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(structured_input)
    structured = tf.keras.layers.BatchNormalization()(structured)
    structured = tf.keras.layers.Dropout(0.20)(structured)

    text_input = tf.keras.Input(shape=(text_dim,), name="text_input")
    text = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(text_input)
    text = tf.keras.layers.Dropout(0.25)(text)
    branches, inputs = [structured, text], [structured_input, text_input]

    if use_sequence:
        sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
        sequence = fresh_sequence_layer(sequence_vocabulary)(sequence_input)
        sequence = tf.keras.layers.Embedding(len(sequence_vocabulary), 16, mask_zero=True)(sequence)
        sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(sequence)
        sequence = tf.keras.layers.Dropout(0.20)(sequence)
        branches.append(sequence)
        inputs.append(sequence_input)

    merged = tf.keras.layers.Concatenate()(branches)
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(0.30)(merged)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(merged)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    if config["loss"] == "focal":
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, alpha=config["alpha"], gamma=config["gamma"]
        )
    else:
        loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=loss)
    return model


def make_task_data(target):
    task = df.dropna(subset=[target]).copy().sort_values("created_at").reset_index(drop=True)
    task[target] = task[target].astype(int)
    n = len(task)
    return task.iloc[:int(n * .70)].copy(), task.iloc[int(n * .70):int(n * .85)].copy(), task.iloc[int(n * .85):].copy()


def create_inputs(train, valid, test, features, use_sequence):
    processor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), features),
        ("source", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    struct_train = processor.fit_transform(train[features + CATEGORICAL]).astype("float32")
    struct_valid = processor.transform(valid[features + CATEGORICAL]).astype("float32")
    struct_test = processor.transform(test[features + CATEGORICAL]).astype("float32")

    def embed(split):
        return text_encoder.encode(split["issue_text_at_cutoff"].tolist(), batch_size=64,
                                   show_progress_bar=True, normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")
    text_train, text_valid, text_test = embed(train), embed(valid), embed(test)

    def make(split, structured, text):
        values = {"structured_input": structured, "text_input": text}
        if use_sequence:
            values["sequence_input"] = split["early_event_sequence"].values.reshape(-1, 1)
        return values
    return (make(train, struct_train, text_train), make(valid, struct_valid, text_valid),
            make(test, struct_test, text_test), struct_train.shape[1], text_train.shape[1])

# %% [markdown]
# ## Train candidate configurations on validation, then evaluate the selected one once on test

# %%
all_results = []
best_models = {}

for target, task_name in TARGETS.items():
    train, valid, test = make_task_data(target)
    timing = approved_timing.get(task_name, [])
    features = BASE_NUMERIC + [feature for feature in timing if feature not in BASE_NUMERIC]
    use_sequence = target != "requirement_volatility_label"
    sequence_vocabulary = sequence_vocabulary_for_train(train) if use_sequence else None
    x_train, x_valid, x_test, structured_dim, text_dim = create_inputs(train, valid, test, features, use_sequence)
    y_train, y_valid, y_test = train[target].to_numpy(), valid[target].to_numpy(), test[target].to_numpy()
    print(f"\n{task_name}: {len(features)} numeric features; sequence={use_sequence}; positives train/test={y_train.sum()}/{y_test.sum()}")

    configs = CONFIGS[target] if RUN_FULL_TUNING else CONFIGS[target][:1]
    candidates = []
    for config in configs:
        model = build_model(structured_dim, text_dim, use_sequence, config, sequence_vocabulary)
        if config["loss"] == "bce":
            ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            weights = np.where(y_train == 1, min(np.sqrt(ratio) * config["positive_scale"], 4.0), 1.0)
        else:
            weights = None
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=1, factor=0.5, min_lr=1e-5, verbose=0),
        ]
        model.fit(x_train, y_train, sample_weight=weights, validation_data=(x_valid, y_valid),
                  epochs=15, batch_size=128, callbacks=callbacks, verbose=0)
        valid_prob = model.predict(x_valid, batch_size=256, verbose=0).reshape(-1)
        threshold = exact_f1_threshold(y_valid, valid_prob)
        valid_f1 = f1_score(y_valid, valid_prob >= threshold, zero_division=0)
        candidates.append((valid_f1, model, config, threshold))
        print(f"  {config['name']}: validation F1={valid_f1:.3f}, threshold={threshold:.3f}")

    _, model, chosen_config, threshold = max(candidates, key=lambda value: value[0])
    test_prob = model.predict(x_test, batch_size=256, verbose=0).reshape(-1)
    test_pred = (test_prob >= threshold).astype(int)
    result = {
        "Task": task_name,
        "Model": "Task-Specific Neural Model v5",
        "Selected loss configuration": chosen_config["name"],
        "Selected timing features": "; ".join(timing),
        "Threshold": threshold,
        "Eligible test records": len(test),
        "Positive test records": int(y_test.sum()),
        "Precision": precision_score(y_test, test_pred, zero_division=0),
        "Recall": recall_score(y_test, test_pred, zero_division=0),
        "F1-score": f1_score(y_test, test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_prob),
        "PR-AUC": average_precision_score(y_test, test_prob),
    }
    all_results.append(result)
    best_models[target] = (model, y_test, test_pred, threshold)

results_df = pd.DataFrame(all_results)
display(results_df.style.format({
    "Threshold": "{:.4f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
    "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}", "PR-AUC": "{:.3f}",
}))
results_df.to_csv("task_specific_neural_results_v5.csv", index=False)
files.download("task_specific_neural_results_v5.csv")

# %%
for target, task_name in TARGETS.items():
    _, y_test, predictions, threshold = best_models[target]
    plt.figure(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, cmap="Blues")
    plt.title(f"{task_name}\nTask-specific v5; threshold={threshold:.3f}")
    plt.tight_layout()
    plt.show()
