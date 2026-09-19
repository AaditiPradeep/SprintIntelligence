"""Create a leakage-safe Jira training dataset from the exported issue records.

For every issue, this script uses only activity observed in the first 30 days
after creation as model input.  It then derives the three target labels from
events after that cut-off.  The raw Jira exports are never changed.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from statistics import quantiles

from build_feature_dataset import (
    DONE_STATUSES,
    REOPEN_STATUSES,
    REQUIREMENT_FIELDS,
    as_list,
    get_name,
    load_json_lines,
    parse_date,
    text_len,
)


INPUTS = [
    ("SecondLife", "exports", "secondlife_5000.json"),
    ("JiraEcosystem", "exports", "jiraecosystem_5000.json"),
    ("IntelDAOS", "exports", "inteldaos_5000.json"),
    ("Hyperledger", "exports", "hyperledger_28146.json"),
]

APACHE_TEST_INPUTS = [
    ("ApacheOldSample", "samples", "apache_500.ndjson"),
    ("ApacheRecentSample", "samples", "apache_recent_500.ndjson"),
]


def history_events(doc, cutoff):
    """Return feature counts before cutoff and outcome counts after cutoff."""
    early = Counter()
    future = Counter()
    for history in as_list(doc.get("changelog", {}).get("histories")):
        history_date = parse_date(history.get("created"))
        if history_date is None:
            continue
        bucket = early if history_date <= cutoff else future
        bucket["history_count"] += 1
        for item in as_list(history.get("items")):
            field = str(item.get("field") or "").strip().lower()
            bucket["item_count"] += 1
            if field == "status":
                bucket["status_change_count"] += 1
                from_status = str(item.get("fromString") or "").strip().lower()
                to_status = str(item.get("toString") or "").strip().lower()
                if from_status in DONE_STATUSES and to_status in REOPEN_STATUSES:
                    bucket["reopen_transition_count"] += 1
            elif field == "assignee":
                bucket["assignee_change_count"] += 1
            elif field == "priority":
                bucket["priority_change_count"] += 1
            elif field == "description":
                bucket["description_change_count"] += 1
            elif field == "summary":
                bucket["summary_change_count"] += 1
            elif field in {"component", "components"}:
                bucket["component_change_count"] += 1
            elif field == "labels":
                bucket["label_change_count"] += 1
            if field in REQUIREMENT_FIELDS:
                bucket["requirement_change_count"] += 1
            if field in {"assignee", "status", "remoteissuelink", "link"}:
                bucket["developer_activity_count"] += 1
    return early, future


def make_row(source, doc, window_days):
    fields = doc.get("fields", {})
    created = parse_date(fields.get("created"))
    updated = parse_date(fields.get("updated"))
    resolved = parse_date(fields.get("resolutiondate"))
    if created is None:
        return None

    cutoff = created + timedelta(days=window_days)
    # An issue with no data beyond the cut-off cannot reliably receive a
    # negative future-event label, so it remains in the CSV but targets are blank.
    observed_after_cutoff = updated is not None and updated > cutoff
    early, future = history_events(doc, cutoff)

    comments_value = fields.get("comments")
    comments_available = isinstance(comments_value, list)
    if not comments_available and isinstance(fields.get("comment"), dict):
        comments_value = fields["comment"].get("comments")
        comments_available = isinstance(comments_value, list)
    comments = as_list(comments_value)
    early_comments = [
        comment for comment in comments
        if isinstance(comment, dict)
        and (comment_date := parse_date(comment.get("created"))) is not None
        and comment_date <= cutoff
    ]
    comment_lengths = [text_len(comment.get("body")) for comment in early_comments]

    row = {
        "source_collection": source,
        "issue_key": doc.get("key", ""),
        "project_key": get_name(fields.get("project", {})) or "Unknown",
        "created_at": created.isoformat(),
        "created_year": created.year,
        "created_month": created.month,
        "observation_window_days": window_days,
        "comments_available": int(comments_available),
        "early_comment_count": len(early_comments),
        "early_avg_comment_length": round(sum(comment_lengths) / len(comment_lengths), 2) if comment_lengths else 0,
        "early_history_count": early["history_count"],
        "early_changelog_item_count": early["item_count"],
        "early_status_change_count": early["status_change_count"],
        "early_assignee_change_count": early["assignee_change_count"],
        "early_priority_change_count": early["priority_change_count"],
        "early_description_change_count": early["description_change_count"],
        "early_summary_change_count": early["summary_change_count"],
        "early_component_change_count": early["component_change_count"],
        "early_label_change_count": early["label_change_count"],
        "early_developer_activity_count": early["developer_activity_count"],
        "observed_after_cutoff": int(observed_after_cutoff),
        "eventual_resolution_days": (resolved - created).days if resolved and resolved >= created else "",
        "future_requirement_change_count": future["requirement_change_count"],
        "future_reopen_transition_count": future["reopen_transition_count"],
    }
    # Targets are purposely constructed AFTER the observation window.
    row["requirement_volatility_label"] = int(future["requirement_change_count"] >= 2) if observed_after_cutoff else ""
    row["issue_reopen_label"] = int(future["reopen_transition_count"] > 0) if observed_after_cutoff else ""
    return row


def add_resolution_labels(rows):
    """Use source-specific 75th percentiles when enough resolved issues exist."""
    durations = defaultdict(list)
    for row in rows:
        duration = row["eventual_resolution_days"]
        if duration != "" and duration > row["observation_window_days"]:
            durations[row["source_collection"]].append(duration)
    all_durations = [duration for values in durations.values() for duration in values]
    global_threshold = quantiles(all_durations, n=4)[2] if len(all_durations) >= 4 else 0
    thresholds = {}
    for source, values in durations.items():
        thresholds[source] = quantiles(values, n=4)[2] if len(values) >= 30 else global_threshold
    for row in rows:
        duration = row["eventual_resolution_days"]
        if duration == "" or duration <= row["observation_window_days"]:
            row["issue_resolution_risk_label"] = ""
        else:
            row["issue_resolution_risk_label"] = int(duration > thresholds[row["source_collection"]])
    return thresholds, global_threshold


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build leakage-safe Jira training dataset (v2).")
    parser.add_argument("--input-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--include-apache-test-samples", action="store_true",
                        help="Include temporary Apache samples. Off by default because they do not contain comments and are not part of the selected final dataset.")
    args = parser.parse_args()

    rows, seen, source_counts = [], set(), Counter()
    inputs = INPUTS + (APACHE_TEST_INPUTS if args.include_apache_test_samples else [])
    for source, folder, filename in inputs:
        path = args.input_dir / folder / filename
        for doc in load_json_lines(path):
            row = make_row(source, doc, args.window_days)
            if row is None or not row["issue_key"]:
                continue
            unique_key = (source, row["issue_key"])
            if unique_key in seen:
                continue
            seen.add(unique_key)
            rows.append(row)
            source_counts[source] += 1
    if not rows:
        raise FileNotFoundError("No issue records found. Check --input-dir.")

    thresholds, global_threshold = add_resolution_labels(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "training_dataset_v2.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # This baseline list intentionally excludes identifiers, timestamps, and
    # every post-cutoff/audit field.  It can be passed directly to the model
    # notebook so future-event columns never slip into model inputs.
    safe_model_features = [
        "source_collection", "created_year", "created_month",
        "comments_available",
        "early_comment_count", "early_avg_comment_length",
        "early_history_count", "early_changelog_item_count",
        "early_status_change_count", "early_assignee_change_count",
        "early_priority_change_count", "early_description_change_count",
        "early_summary_change_count", "early_component_change_count",
        "early_label_change_count", "early_developer_activity_count",
    ]
    (args.output_dir / "model_feature_columns_v2.json").write_text(
        json.dumps(safe_model_features, indent=2), encoding="utf-8"
    )

    targets = ["requirement_volatility_label", "issue_resolution_risk_label", "issue_reopen_label"]
    summary = {
        "dataset": "training_dataset_v2.csv",
        "row_count": len(rows),
        "source_counts": dict(source_counts),
        "observation_window_days": args.window_days,
        "deduplication_key": "source_collection + issue_key",
        "resolution_delay_threshold_days_by_source": thresholds,
        "fallback_global_resolution_delay_threshold_days": global_threshold,
        "target_eligibility_and_positive_counts": {
            target: {
                "eligible_rows": sum(row[target] != "" for row in rows),
                "positive_rows": sum(row[target] == 1 for row in rows),
            }
            for target in targets
        },
        "methodology_note": "Features are limited to the first observation window; labels use only later events or eventual resolution.",
        "safe_model_features_file": "model_feature_columns_v2.json",
    }
    (args.output_dir / "training_dataset_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
