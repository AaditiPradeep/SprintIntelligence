"""Create the clean v6 Jira dataset for Sprint Intelligence.

The observation window remains the first 30 days after issue creation.  v6
changes the weak parts of v5 without changing any raw MongoDB/export data:

* Missing timing values are written as blank/NaN, never as the artificial
  value ``window_days + 1``.  Matching ``has_...`` flags preserve the useful
  distinction between "no event occurred" and an actual timing value.
* ``days_to_first_activity`` is deliberately excluded: issue creation and
  imports make it almost constant and therefore unhelpful.
* Labels use a fixed future horizon (90 days by default), rather than a
  source-wide percentile calculated from all issues.

Only fields available at the observation cutoff are model features.  The
future-window audit columns and labels must never be used as model inputs.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

from build_feature_dataset import DONE_STATUSES, REOPEN_STATUSES, REQUIREMENT_FIELDS, as_list, load_json_lines, parse_date
from build_training_dataset_v2 import INPUTS, make_row
from build_training_dataset_v3 import add_multimodal_fields, clean_text


def comments_from_doc(doc):
    fields = doc.get("fields", {})
    comments = fields.get("comments")
    if comments is None and isinstance(fields.get("comment"), dict):
        comments = fields["comment"].get("comments")
    return as_list(comments)


def v6_timing_features(doc, created, cutoff, window_days):
    """Return cutoff-safe timing fields with missing values plus availability flags."""
    comment_times, comment_authors = [], set()
    for comment in comments_from_doc(doc):
        if not isinstance(comment, dict):
            continue
        event_time = parse_date(comment.get("created"))
        if event_time is not None and event_time <= cutoff:
            comment_times.append(event_time)
            author = comment.get("author") if isinstance(comment.get("author"), dict) else {}
            author_id = author.get("key") or author.get("name")
            if author_id:
                comment_authors.add(str(author_id))

    history_times, assignee_times, status_times = [], [], []
    history_authors, assignee_values = set(), set()
    item_count = 0
    for history in as_list(doc.get("changelog", {}).get("histories")):
        event_time = parse_date(history.get("created"))
        if event_time is None or event_time > cutoff:
            continue
        history_times.append(event_time)
        author = history.get("author") if isinstance(history.get("author"), dict) else {}
        author_id = author.get("key") or author.get("name")
        if author_id:
            history_authors.add(str(author_id))
        for item in as_list(history.get("items")):
            item_count += 1
            field = str(item.get("field") or "").strip().lower()
            if field == "assignee":
                assignee_times.append(event_time)
                if clean_text(item.get("toString")):
                    assignee_values.add(clean_text(item.get("toString")))
            elif field == "status":
                status_times.append(event_time)

    def timing_value(times):
        if not times:
            return ""
        return round(max(0, (min(times) - created).total_seconds() / 86400), 2)

    activity_times = sorted(comment_times + history_times)
    timeline = [created] + activity_times + [cutoff]
    gaps = [max(0, (later - earlier).total_seconds() / 86400) for earlier, later in zip(timeline, timeline[1:])]
    return {
        "has_early_comment": int(bool(comment_times)),
        "has_early_assignee_change": int(bool(assignee_times)),
        "has_early_status_change": int(bool(status_times)),
        "days_to_first_comment": timing_value(comment_times),
        "days_to_first_assignee_change": timing_value(assignee_times),
        "days_to_first_status_change": timing_value(status_times),
        "max_inactivity_gap_days": round(max(gaps) if gaps else window_days, 2),
        "early_active_days": len({event.date() for event in activity_times}),
        "early_events_per_day": round(item_count / window_days, 4),
        "early_comments_per_day": round(len(comment_times) / window_days, 4),
        "unique_early_assignees": len(assignee_values),
        "unique_early_commenters": len(comment_authors),
        "unique_early_changelog_authors": len(history_authors),
    }


def latest_observed_time(doc):
    """Latest timestamp present in the static issue export."""
    fields = doc.get("fields", {})
    candidates = [parse_date(fields.get("updated")), parse_date(fields.get("resolutiondate"))]
    candidates.extend(parse_date(history.get("created")) for history in as_list(doc.get("changelog", {}).get("histories")) if isinstance(history, dict))
    candidates.extend(parse_date(comment.get("created")) for comment in comments_from_doc(doc) if isinstance(comment, dict))
    return max((value for value in candidates if value is not None), default=None)


def future_label_events(doc, cutoff, label_end):
    requirement_changes = 0
    done_times, reopen_times = [], []
    for history in as_list(doc.get("changelog", {}).get("histories")):
        if not isinstance(history, dict):
            continue
        event_time = parse_date(history.get("created"))
        if event_time is None or event_time <= cutoff:
            continue
        for item in as_list(history.get("items")):
            field = str(item.get("field") or "").strip().lower()
            if cutoff < event_time <= label_end and field in REQUIREMENT_FIELDS:
                requirement_changes += 1
            if field == "status":
                from_status = str(item.get("fromString") or "").strip().lower()
                to_status = str(item.get("toString") or "").strip().lower()
                if to_status in DONE_STATUSES:
                    done_times.append(event_time)
                if from_status in DONE_STATUSES and to_status in REOPEN_STATUSES:
                    reopen_times.append(event_time)
    return requirement_changes, sorted(done_times), sorted(reopen_times)


def add_v6_labels(row, doc, window_days, horizon_days, volatility_change_threshold):
    """Create horizon-based labels and preserve eligibility information for audit."""
    created = parse_date(doc.get("fields", {}).get("created"))
    cutoff = created + timedelta(days=window_days)
    label_end = cutoff + timedelta(days=horizon_days)
    observed_until = latest_observed_time(doc)
    resolution_date = parse_date(doc.get("fields", {}).get("resolutiondate"))
    requirement_changes, done_times, reopen_times = future_label_events(doc, cutoff, label_end)

    # Volatility is observable if the issue reached the end of the horizon, or
    # it was resolved before then (after resolution there are no further scope
    # changes for this issue in the data).
    volatility_eligible = bool(observed_until and observed_until >= label_end) or bool(resolution_date and cutoff < resolution_date <= label_end)
    row["future_requirement_change_count_90d"] = requirement_changes
    row["requirement_volatility_label"] = int(requirement_changes >= volatility_change_threshold) if volatility_eligible else ""
    row["requirement_volatility_label_eligible"] = int(volatility_eligible)

    # Delay means unresolved at the end of the fixed future horizon, or
    # resolved after it.  Issues whose export ends before the horizon remain
    # unlabelled rather than being guessed as negatives.
    # Issues resolved inside the observation window are already complete at
    # prediction time and do not belong to this task's candidate population.
    resolution_eligible = bool(resolution_date and resolution_date > cutoff) or bool(resolution_date is None and observed_until and observed_until >= label_end)
    if resolution_eligible:
        row["issue_resolution_risk_label"] = int(resolution_date is None or resolution_date > label_end)
    else:
        row["issue_resolution_risk_label"] = ""
    row["issue_resolution_risk_label_eligible"] = int(resolution_eligible)
    row["resolution_deadline_days_from_creation"] = window_days + horizon_days

    # Reopen is meaningful only after a post-cutoff resolution.  A negative is
    # assigned only once its full 90-day post-resolution period is observable.
    first_resolution = min(done_times) if done_times else (resolution_date if resolution_date and resolution_date > cutoff else None)
    reopen_deadline = first_resolution + timedelta(days=horizon_days) if first_resolution else None
    reopen_within_horizon = bool(first_resolution and any(first_resolution < event <= reopen_deadline for event in reopen_times))
    reopen_eligible = bool(first_resolution and (reopen_within_horizon or (observed_until and observed_until >= reopen_deadline)))
    row["first_resolution_at_or_after_cutoff"] = first_resolution.isoformat() if first_resolution else ""
    row["reopen_label_horizon_days"] = horizon_days
    row["issue_reopen_label"] = int(reopen_within_horizon) if reopen_eligible else ""
    row["issue_reopen_label_eligible"] = int(reopen_eligible)
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the clean, horizon-labelled Jira training dataset (v6).")
    parser.add_argument("--input-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--label-horizon-days", type=int, default=90)
    parser.add_argument("--volatility-change-threshold", type=int, default=2)
    parser.add_argument("--output-name", default="training_dataset_v6.csv")
    args = parser.parse_args()

    rows, seen, source_counts = [], set(), Counter()
    for source, folder, filename in INPUTS:
        for doc in load_json_lines(args.input_dir / folder / filename):
            row = make_row(source, doc, args.window_days)
            if row is None or not row["issue_key"]:
                continue
            identity = (source, row["issue_key"])
            if identity in seen:
                continue
            seen.add(identity)
            row = add_multimodal_fields(row, doc, args.window_days)
            row.update(v6_timing_features(doc, parse_date(doc["fields"]["created"]), parse_date(doc["fields"]["created"]) + timedelta(days=args.window_days), args.window_days))
            row.pop("days_to_first_activity", None)
            rows.append(add_v6_labels(row, doc, args.window_days, args.label_horizon_days, args.volatility_change_threshold))
            source_counts[source] += 1
    if not rows:
        raise FileNotFoundError("No selected Public Jira records were found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_name
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    safe_features = [
        "source_collection", "created_year", "created_month", "comments_available",
        "early_comment_count", "early_avg_comment_length", "early_history_count",
        "early_changelog_item_count", "early_status_change_count", "early_assignee_change_count",
        "early_priority_change_count", "early_description_change_count", "early_summary_change_count",
        "early_component_change_count", "early_label_change_count", "early_developer_activity_count",
        "priority_at_cutoff", "issue_type_at_cutoff", "status_at_cutoff",
        "assignee_available_at_cutoff", "text_character_count_at_cutoff",
        "early_event_sequence_length",
        "has_early_comment", "has_early_assignee_change", "has_early_status_change",
        "days_to_first_comment", "days_to_first_assignee_change", "days_to_first_status_change",
        "max_inactivity_gap_days", "early_active_days", "early_events_per_day",
        "early_comments_per_day", "unique_early_assignees", "unique_early_commenters",
        "unique_early_changelog_authors",
    ]
    feature_file = args.output_dir / "model_feature_columns_v6.json"
    feature_file.write_text(json.dumps(safe_features, indent=2), encoding="utf-8")
    targets = ["requirement_volatility_label", "issue_resolution_risk_label", "issue_reopen_label"]
    summary = {
        "dataset": output.name,
        "row_count": len(rows),
        "source_counts": dict(source_counts),
        "observation_window_days": args.window_days,
        "label_horizon_days": args.label_horizon_days,
        "resolution_deadline_days_from_creation": args.window_days + args.label_horizon_days,
        "volatility_change_threshold_within_horizon": args.volatility_change_threshold,
        "target_eligibility_and_positive_counts": {
            target: {"eligible_rows": sum(row[target] != "" for row in rows), "positive_rows": sum(row[target] == 1 for row in rows)}
            for target in targets
        },
        "safe_model_features_file": feature_file.name,
        "methodology_note": "v6 uses cutoff-safe features, blank timing values plus availability flags, and fixed future horizons. Future audit and label fields must never be model inputs.",
    }
    summary_name = Path(args.output_name).stem + "_summary.json"
    (args.output_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
