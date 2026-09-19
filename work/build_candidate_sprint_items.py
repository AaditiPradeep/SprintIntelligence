"""Build a model-ready candidate backlog file from a new Jira export.

This script never changes the MongoDB export.  It reconstructs only the
fields available in the first 30 days after each issue was created and writes
the v6.1 model-feature schema expected by the selected model artifacts.
"""

import argparse
import csv
import json
import sys
from datetime import timedelta
from pathlib import Path

from build_feature_dataset import load_json_lines, parse_date
from build_training_dataset_v2 import make_row
from build_training_dataset_v3 import add_multimodal_fields
from build_training_dataset_v6 import v6_timing_features


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create a v6.1 candidate sprint-item CSV from a Jira JSON-lines export.")
    parser.add_argument("--input", type=Path, default=root / "data" / "samples" / "apache_recent_500.ndjson")
    parser.add_argument("--source", default="Apache")
    parser.add_argument("--output", type=Path, default=root / "data" / "processed" / "candidate_sprint_items.csv")
    parser.add_argument("--feature-file", type=Path, default=root / "data" / "processed" / "model_feature_columns_v6.json")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    feature_columns = json.loads(args.feature_file.read_text(encoding="utf-8"))
    rows, seen = [], set()
    for doc in load_json_lines(args.input):
        row = make_row(args.source, doc, args.window_days)
        if row is None or not row["issue_key"] or row["issue_key"] in seen:
            continue
        fields = doc.get("fields", {})
        created = parse_date(fields.get("created"))
        if created is None:
            continue
        cutoff = created + timedelta(days=args.window_days)
        row = add_multimodal_fields(row, doc, args.window_days)
        row.update(v6_timing_features(doc, created, cutoff, args.window_days))
        row.pop("days_to_first_activity", None)

        # A completed issue is not a candidate for a *future* sprint.  Status
        # is reconstructed at the observation cutoff by add_multimodal_fields.
        if str(row.get("status_at_cutoff", "")).strip().lower() in {"done", "resolved", "closed", "complete", "completed", "fixed"}:
            continue
        feature_values = {name: row.get(name, "") for name in feature_columns if name != "source_collection"}
        rows.append({"source_collection": args.source, "issue_key": row["issue_key"], "created_at": row["created_at"], **feature_values})
        seen.add(row["issue_key"])

    rows.sort(key=lambda item: item["created_at"], reverse=True)
    rows = rows[:args.limit]
    if not rows:
        raise ValueError("No eligible non-complete candidate issues were found in the export.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_collection", "issue_key", "created_at"] + [name for name in feature_columns if name != "source_collection"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "candidate_file": args.output.name,
        "raw_export": str(args.input),
        "source_collection": args.source,
        "candidate_item_count": len(rows),
        "observation_window_days": args.window_days,
        "selection_rule": "Most-recent non-complete issues at the 30-day observation cutoff.",
        "important_limitation": "Apache is unseen by the trained models. Its source category is handled as unknown by the encoder, so these cross-source scores are a dashboard demonstration rather than a validated deployment result.",
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
