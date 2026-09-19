# Sprint Intelligence — multi-risk aggregation and Sprint Health Score (Colab)
# Upload the three selected-model .joblib files. Also upload either:
# 1) candidate_sprint_items.csv (real candidate backlog items with the v6.1
#    feature columns), or 2) training_dataset_v6_1.csv for a demonstration.

# %%
from google.colab import files
uploaded = files.upload()

# %%
import io
import json
import joblib
import numpy as np
import pandas as pd

VOL_MODEL = "requirement_volatility_selected_model_v6_1.joblib"
RES_MODEL = "issue_resolution_risk_selected_model_v6_1.joblib"
REOPEN_MODEL = "issue_reopen_selected_model_v6_1.joblib"
DEMO_DATASET = "training_dataset_v6_1.csv"
CANDIDATE_FILE = "candidate_sprint_items.csv"

# These are transparent business weights, not learned probabilities. Resolution
# receives the largest weight because it has the strongest model and affects
# sprint delivery directly. They can be adjusted with supervisor approval.
WEIGHTS = {"volatility": 0.25, "resolution": 0.50, "reopen": 0.25}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# Use a real candidate file when available. The fallback is only for demo.
if CANDIDATE_FILE in uploaded:
    candidates = pd.read_csv(io.BytesIO(uploaded[CANDIDATE_FILE]), low_memory=False)
    input_mode = "candidate backlog file"
else:
    raw = pd.read_csv(io.BytesIO(uploaded[DEMO_DATASET]), low_memory=False)
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, errors="coerce", format="mixed")
    # Demo backlog: most recent 30 Hyperledger issues. These labels are never
    # passed to a model; use a real candidate_sprint_items.csv for deployment.
    candidates = raw.loc[raw["source_collection"] == "Hyperledger"].sort_values("created_at", ascending=False).head(30).copy()
    input_mode = "demo subset of historical data"

for column in ["source_collection", "issue_key"]:
    if column not in candidates.columns:
        raise ValueError(f"Candidate input is missing required column: {column}")

vol_artifact = joblib.load(io.BytesIO(uploaded[VOL_MODEL]))
res_artifact = joblib.load(io.BytesIO(uploaded[RES_MODEL]))
reopen_artifact = joblib.load(io.BytesIO(uploaded[REOPEN_MODEL]))
print(f"Scoring {len(candidates):,} items using {input_mode}.")

# %%
def risk_probability(artifact, frame):
    required = artifact["numeric_features"] + artifact["categorical_features"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Candidate input lacks model features: " + ", ".join(missing))
    transformed = artifact["processor"].transform(frame[required])
    selected = artifact["selector"].transform(transformed)
    return artifact["model"].predict_proba(selected)[:, 1]

def action_for(row):
    # Priority order prevents contradictory recommendations.
    if row["resolution_risk"] >= 0.70 or row["combined_risk"] >= 0.70:
        return "Defer", "High delivery-delay risk exceeds the sprint-planning tolerance."
    if row["volatility_risk"] >= 0.60:
        return "Split", "High requirement-volatility risk suggests reducing scope before commitment."
    if row["reopen_risk"] >= 0.45:
        return "Monitor", "Elevated post-resolution rework risk needs test and review checkpoints."
    return "Include", "Combined risk is within the current sprint-planning tolerance."

# %%
risk_register = candidates[["source_collection", "issue_key"]].copy()
risk_register["volatility_risk"] = risk_probability(vol_artifact, candidates)
risk_register["resolution_risk"] = risk_probability(res_artifact, candidates)
risk_register["reopen_risk"] = risk_probability(reopen_artifact, candidates)
risk_register["combined_risk"] = (
    WEIGHTS["volatility"] * risk_register["volatility_risk"]
    + WEIGHTS["resolution"] * risk_register["resolution_risk"]
    + WEIGHTS["reopen"] * risk_register["reopen_risk"]
)
risk_register["risk_band"] = pd.cut(
    risk_register["combined_risk"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
)
actions = risk_register.apply(action_for, axis=1, result_type="expand")
risk_register["planner_action"] = actions[0]
risk_register["action_reason"] = actions[1]
risk_register = risk_register.sort_values("combined_risk", ascending=False).reset_index(drop=True)

# Health is a transparent heuristic: 70% average combined risk and 30% share
# of high-risk items. It is a planning indicator, not a clinical/financial score.
mean_risk = float(risk_register["combined_risk"].mean())
high_risk_share = float((risk_register["combined_risk"] >= 0.70).mean())
health_score = round(100 * (1 - (0.70 * mean_risk + 0.30 * high_risk_share)), 1)
health_band = "Healthy" if health_score >= 70 else "At Risk" if health_score >= 45 else "Critical"

summary = {
    "input_mode": input_mode,
    "candidate_item_count": int(len(risk_register)),
    "risk_weights": WEIGHTS,
    "mean_combined_risk": round(mean_risk, 4),
    "high_risk_item_share": round(high_risk_share, 4),
    "sprint_health_score_out_of_100": health_score,
    "sprint_health_band": health_band,
    "action_counts": risk_register["planner_action"].value_counts().to_dict(),
    "methodology_note": "Health Score is a transparent planning heuristic built from model probabilities. It must be recalculated for each proposed sprint.",
}

display(risk_register.head(20))
print(json.dumps(summary, indent=2))

# %%
risk_register.to_csv("sprint_risk_register_v6_1.csv", index=False)
with open("sprint_health_summary_v6_1.json", "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
with open("risk_aggregation_config_v6_1.json", "w", encoding="utf-8") as handle:
    json.dump({"weights": WEIGHTS, "high_risk_cutoff": 0.70, "medium_risk_cutoff": 0.40}, handle, indent=2)
files.download("sprint_risk_register_v6_1.csv")
files.download("sprint_health_summary_v6_1.json")
files.download("risk_aggregation_config_v6_1.json")
