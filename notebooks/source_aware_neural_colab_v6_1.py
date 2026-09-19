# Sprint Intelligence — v6.1 source-aware neural comparison (Google Colab)
# Run the baseline notebook first. Upload its source_aware_splits_v6_1.csv
# here, together with training_dataset_v6_1.csv and model_feature_columns_v6.json.

# %%
!pip -q install sentence-transformers

# %%
from google.colab import files
uploaded = files.upload()

# %%
import io, json, random, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE, TOP_K_FEATURES = 42, 30
np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.keras.utils.set_random_seed(RANDOM_STATE)
DATA_FILE, FEATURE_FILE, SPLIT_FILE = "training_dataset_v6_1.csv", "model_feature_columns_v6.json", "source_aware_splits_v6_1_fixed.csv"
df = pd.read_csv(io.BytesIO(uploaded[DATA_FILE]), low_memory=False)
safe_features = json.loads(uploaded[FEATURE_FILE].decode("utf-8"))
split_map = pd.read_csv(io.BytesIO(uploaded[SPLIT_FILE]))
# Match the baseline notebook: Public Jira timestamps use mixed valid formats.
df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce", format="mixed")
df = df.dropna(subset=["created_at", "source_collection", "issue_key"]).copy()
df["issue_text_at_cutoff"] = df["issue_text_at_cutoff"].fillna("").astype(str).str.slice(0, 3000)
df["early_event_sequence"] = df["early_event_sequence"].fillna("").astype(str)

TARGETS = {
    "requirement_volatility_label": "Requirement Volatility Risk",
    "issue_resolution_risk_label": "Issue Resolution Risk",
    "issue_reopen_label": "Issue Reopen Risk",
}
CATEGORICAL = ["source_collection", "priority_at_cutoff", "issue_type_at_cutoff", "status_at_cutoff"]
NUMERIC = [column for column in safe_features if column not in CATEGORICAL]
INCLUDE_SOURCE_FEATURE = True  # Keep identical to the main baseline run.
if not INCLUDE_SOURCE_FEATURE:
    CATEGORICAL.remove("source_collection")
print(f"Dataset rows: {len(df):,}; split rows: {len(split_map):,}")

# %%
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

def sequence_vocabulary(train):
    layer = tf.keras.layers.TextVectorization(max_tokens=200, output_mode="int", output_sequence_length=80, standardize=None, split="whitespace")
    layer.adapt(tf.data.Dataset.from_tensor_slices(train["early_event_sequence"].values).batch(256))
    return layer.get_vocabulary()

def sequence_layer(vocabulary):
    layer = tf.keras.layers.TextVectorization(max_tokens=200, output_mode="int", output_sequence_length=80, standardize=None, split="whitespace")
    layer.set_vocabulary(vocabulary)
    return layer

def build_model(structured_dim, text_dim, use_sequence, vocabulary, loss_name):
    tf.keras.backend.clear_session()
    structured_input = tf.keras.Input(shape=(structured_dim,), name="structured_input")
    structured = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(structured_input)
    structured = tf.keras.layers.BatchNormalization()(structured)
    structured = tf.keras.layers.Dropout(.30)(structured)
    text_input = tf.keras.Input(shape=(text_dim,), name="text_input")
    text = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(text_input)
    text = tf.keras.layers.Dropout(.25)(text)
    branches, inputs = [structured, text], [structured_input, text_input]
    if use_sequence:
        sequence_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="sequence_input")
        sequence = sequence_layer(vocabulary)(sequence_input)
        sequence = tf.keras.layers.Embedding(len(vocabulary), 16, mask_zero=True)(sequence)
        sequence = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(sequence)
        sequence = tf.keras.layers.Dropout(.25)(sequence)
        branches.append(sequence); inputs.append(sequence_input)
    merged = tf.keras.layers.Concatenate()(branches)
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)
    merged = tf.keras.layers.Dropout(.35)(merged)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(merged)
    loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, alpha=.75, gamma=2.0) if loss_name == "focal" else tf.keras.losses.BinaryCrossentropy()
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), loss=loss, metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def task_partitions(target):
    assignments = split_map.loc[split_map["target"] == target, ["source_collection", "issue_key", "split"]]
    task = df.dropna(subset=[target]).copy()
    task[target] = task[target].astype(int)
    task = task.merge(assignments, on=["source_collection", "issue_key"], how="inner", validate="one_to_one")
    if len(task) != len(assignments):
        raise ValueError(f"Split file and dataset do not match for {target}.")
    return (task.loc[task["split"] == name].copy() for name in ["train", "validation", "test"])

def model_inputs(train, valid, test, use_sequence):
    processor = make_preprocessor()
    x_train = processor.fit_transform(train[NUMERIC + CATEGORICAL]).astype("float32")
    x_valid = processor.transform(valid[NUMERIC + CATEGORICAL]).astype("float32")
    x_test = processor.transform(test[NUMERIC + CATEGORICAL]).astype("float32")
    selector = SelectKBest(mutual_info_classif, k=min(TOP_K_FEATURES, x_train.shape[1])).fit(x_train, train[CURRENT_TARGET].to_numpy())
    x_train, x_valid, x_test = selector.transform(x_train), selector.transform(x_valid), selector.transform(x_test)
    def embed(frame):
        return text_encoder.encode(frame["issue_text_at_cutoff"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    text_train, text_valid, text_test = embed(train), embed(valid), embed(test)
    vocabulary = sequence_vocabulary(train) if use_sequence else None
    def pack(frame, structured, text):
        values = {"structured_input": structured, "text_input": text}
        if use_sequence:
            values["sequence_input"] = tf.reshape(tf.constant(frame["early_event_sequence"].to_numpy(), dtype=tf.string), (-1, 1))
        return values
    return pack(train, x_train, text_train), pack(valid, x_valid, text_valid), pack(test, x_test, text_test), x_train.shape[1], text_train.shape[1], vocabulary

# %%
all_results = []
for CURRENT_TARGET, task_name in TARGETS.items():
    train, valid, test = task_partitions(CURRENT_TARGET)
    y_train, y_valid, y_test = train[CURRENT_TARGET].to_numpy(), valid[CURRENT_TARGET].to_numpy(), test[CURRENT_TARGET].to_numpy()
    use_sequence = CURRENT_TARGET != "requirement_volatility_label"
    x_train, x_valid, x_test, structured_dim, text_dim, vocabulary = model_inputs(train, valid, test, use_sequence)
    configs = ["focal"] if CURRENT_TARGET == "issue_reopen_label" else ["bce", "bce_weighted"]
    candidates = []
    print(f"\n{task_name}: train/validation/test={len(train):,}/{len(valid):,}/{len(test):,}; positives={y_train.sum()}/{y_valid.sum()}/{y_test.sum()}")
    for config in configs:
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        model = build_model(structured_dim, text_dim, use_sequence, vocabulary, "focal" if config == "focal" else "bce")
        sample_weight = None
        if config == "bce_weighted":
            ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            sample_weight = np.where(y_train == 1, min(np.sqrt(ratio), 4.0), 1.0)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=2, factor=.5, min_lr=1e-5, verbose=0),
        ]
        model.fit(x_train, y_train, sample_weight=sample_weight, validation_data=(x_valid, y_valid), epochs=25, batch_size=128, callbacks=callbacks, verbose=0)
        valid_probability = model.predict(x_valid, batch_size=256, verbose=0).reshape(-1)
        test_probability = model.predict(x_test, batch_size=256, verbose=0).reshape(-1)
        threshold = choose_threshold(y_valid, valid_probability)  # Validation only.
        valid_result, test_result = metrics(y_valid, valid_probability, threshold), metrics(y_test, test_probability, threshold)
        criterion = valid_result["PR-AUC"] if CURRENT_TARGET == "issue_reopen_label" else valid_result["F1-score"]
        candidates.append((criterion, config, threshold, test_result, valid_result))
        print(f"  {config}: validation F1={valid_result['F1-score']:.3f}, PR-AUC={valid_result['PR-AUC']:.3f}")
    _, config, threshold, test_result, valid_result = max(candidates, key=lambda item: item[0])
    all_results.append({"Task": task_name, "Model": "Task-specific multimodal neural model", "Selected configuration": config, "Threshold (validation selected)": threshold, "Eligible test records": len(test), "Positive test records": int(y_test.sum()), "Validation F1": valid_result["F1-score"], "Validation PR-AUC": valid_result["PR-AUC"], **test_result})

# %%
