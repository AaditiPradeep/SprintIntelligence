"""Create the multimodal, leakage-safe Jira training dataset (v3).

v3 retains every v2 structured feature and target, then adds:
* issue_text_at_cutoff: summary, description and comments available by day 30
* early_event_sequence: ordered workflow events available by day 30

Text fields changed after the cutoff are reconstructed from Jira history when
possible; otherwise that field is omitted rather than leaking future text.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from build_feature_dataset import as_list, get_name, load_json_lines, parse_date
from build_training_dataset_v2 import INPUTS, add_resolution_labels, make_row


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def token(value):
    value = clean_text(value).upper()
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    return value[:40] or "UNKNOWN"


def value_at_cutoff(doc, field_name, cutoff):
    """Reconstruct a current text field back to its value at the cutoff.

    Jira histories hold the value before a change in fromString. If that value
    is missing for a post-cutoff edit, return an empty value to avoid leaking
    the current/future text.
    """
    value = doc.get("fields", {}).get(field_name)
    if not isinstance(value, str):
        return "", "not_available"
    histories = []
    for history in as_list(doc.get("changelog", {}).get("histories")):
        event_time = parse_date(history.get("created"))
        if event_time is not None and event_time > cutoff:
            histories.append((event_time, history))
    changed_after_cutoff = False
    for _, history in sorted(histories, key=lambda item: item[0], reverse=True):
        for item in as_list(history.get("items")):
            if str(item.get("field") or "").strip().lower() != field_name:
                continue
            changed_after_cutoff = True
            previous_value = item.get("fromString")
            if not isinstance(previous_value, str):
                return "", "omitted_after_unrecoverable_change"
            value = previous_value
    return clean_text(value), "reconstructed" if changed_after_cutoff else "unchanged"


def scalar_at_cutoff(doc, field_name, history_field_names, cutoff):
    """Safely reconstruct a categorical Jira field at the observation cutoff."""
    value = get_name(doc.get("fields", {}).get(field_name))
    changed_after_cutoff = False
    histories = []
    for history in as_list(doc.get("changelog", {}).get("histories")):
        event_time = parse_date(history.get("created"))
        if event_time is not None and event_time > cutoff:
            histories.append((event_time, history))
    for _, history in sorted(histories, key=lambda item: item[0], reverse=True):
        for item in as_list(history.get("items")):
            field = str(item.get("field") or "").strip().lower()
            if field not in history_field_names:
                continue
            changed_after_cutoff = True
            previous_value = item.get("fromString")
            if not isinstance(previous_value, str) or not previous_value.strip():
                return "Unknown", "omitted_after_unrecoverable_change"
            value = previous_value
    value = clean_text(value) or "Unknown"
    return value, "reconstructed" if changed_after_cutoff else "unchanged"


def assignee_available_at_cutoff(doc, cutoff):
    """Reverse later assignment changes to determine whether an owner existed at cutoff."""
    assigned = bool(doc.get("fields", {}).get("assignee"))
    histories = []
    for history in as_list(doc.get("changelog", {}).get("histories")):
        event_time = parse_date(history.get("created"))
        if event_time is not None and event_time > cutoff:
            histories.append((event_time, history))
    changed_after_cutoff = False
    for _, history in sorted(histories, key=lambda item: item[0], reverse=True):
        for item in as_list(history.get("items")):
            if str(item.get("field") or "").strip().lower() != "assignee":
                continue
            changed_after_cutoff = True
            assigned = bool(clean_text(item.get("fromString")))
    return int(assigned), "reconstructed" if changed_after_cutoff else "unchanged"


def early_comments_and_sequence(doc, cutoff):
    fields = doc.get("fields", {})
    comments = fields.get("comments")
    if comments is None and isinstance(fields.get("comment"), dict):
        comments = fields["comment"].get("comments")
    early_comments = []
    for comment in as_list(comments):
        if not isinstance(comment, dict):
            continue
        comment_time = parse_date(comment.get("created"))
        if comment_time is not None and comment_time <= cutoff:
            body = clean_text(comment.get("body"))
            if body:
                early_comments.append((comment_time, body))

    events = []
    for history in as_list(doc.get("changelog", {}).get("histories")):
        event_time = parse_date(history.get("created"))
        if event_time is None or event_time > cutoff:
            continue
        for item in as_list(history.get("items")):
            field = token(item.get("field"))
            if field == "STATUS":
                events.append((event_time, f"STATUS_{token(item.get('fromString'))}_TO_{token(item.get('toString'))}"))
            else:
                events.append((event_time, f"{field}_CHANGE"))
    events.sort(key=lambda event: event[0])
    early_comments.sort(key=lambda comment: comment[0])
    return [body for _, body in early_comments], [event for _, event in events]


def early_timing_features(doc, created, cutoff, window_days):
    """Derive timing and velocity signals only from activity before cutoff."""
    fields = doc.get("fields", {})
    comments = fields.get("comments")
    if comments is None and isinstance(fields.get("comment"), dict):
        comments = fields["comment"].get("comments")

    comment_times, comment_authors = [], set()
    for comment in as_list(comments):
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
                assignee = item.get("toString")
                if assignee:
                    assignee_values.add(str(assignee))
            elif field == "status":
                status_times.append(event_time)

    activity_times = sorted(comment_times + history_times)
    def first_day(times):
        return round((min(times) - created).total_seconds() / 86400, 2) if times else window_days + 1

    # Include creation and cutoff in the gap calculation. With no activity, the
    # maximum inactive period correctly becomes the full observation window.
    timeline = [created] + activity_times + [cutoff]
    gaps = [
        max(0, (later - earlier).total_seconds() / 86400)
        for earlier, later in zip(timeline, timeline[1:])
    ]
    active_days = len({event.date() for event in activity_times})
    return {
        "days_to_first_comment": first_day(comment_times),
        "days_to_first_assignee_change": first_day(assignee_times),
        "days_to_first_status_change": first_day(status_times),
        "days_to_first_activity": first_day(activity_times),
        "max_inactivity_gap_days": round(max(gaps) if gaps else window_days, 2),
        "early_active_days": active_days,
        "early_events_per_day": round(item_count / window_days, 4),
        "early_comments_per_day": round(len(comment_times) / window_days, 4),
        "unique_early_assignees": len(assignee_values),
        "unique_early_commenters": len(comment_authors),
        "unique_early_changelog_authors": len(history_authors),
    }


def add_multimodal_fields(row, doc, window_days):
    created = parse_date(doc.get("fields", {}).get("created"))
    cutoff = created + __import__("datetime").timedelta(days=window_days)
    summary, summary_status = value_at_cutoff(doc, "summary", cutoff)
    description, description_status = value_at_cutoff(doc, "description", cutoff)
    priority, priority_status = scalar_at_cutoff(doc, "priority", {"priority"}, cutoff)
    issue_type, issue_type_status = scalar_at_cutoff(doc, "issuetype", {"issuetype", "issue type"}, cutoff)
    status, status_snapshot_status = scalar_at_cutoff(doc, "status", {"status"}, cutoff)
    assignee_available, assignee_snapshot_status = assignee_available_at_cutoff(doc, cutoff)
    comment_texts, sequence = early_comments_and_sequence(doc, cutoff)
    timing = early_timing_features(doc, created, cutoff, window_days)
    text_sections = []
    if summary:
        text_sections.append("SUMMARY: " + summary)
    if description:
        text_sections.append("DESCRIPTION: " + description)
    if comment_texts:
        text_sections.append("EARLY_COMMENTS: " + " ".join(comment_texts))
    row.update({
        "issue_text_at_cutoff": " ".join(text_sections),
        "text_character_count_at_cutoff": len(" ".join(text_sections)),
        "summary_snapshot_status": summary_status,
        "description_snapshot_status": description_status,
        "priority_at_cutoff": priority,
        "issue_type_at_cutoff": issue_type,
        "status_at_cutoff": status,
        "assignee_available_at_cutoff": assignee_available,
        "priority_snapshot_status": priority_status,
        "issue_type_snapshot_status": issue_type_status,
        "status_snapshot_status": status_snapshot_status,
        "assignee_snapshot_status": assignee_snapshot_status,
        "early_event_sequence": " ".join(sequence),
        "early_event_sequence_length": len(sequence),
    })
    row.update(timing)
    return row


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build multimodal Jira training dataset (v3).")
    parser.add_argument("--input-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--output-name", default="training_dataset_v3.csv")
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
            rows.append(add_multimodal_fields(row, doc, args.window_days))
            source_counts[source] += 1
    if not rows:
        raise FileNotFoundError("No selected Public Jira records were found.")

    thresholds, global_threshold = add_resolution_labels(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_name
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": output.name,
        "row_count": len(rows),
        "source_counts": dict(source_counts),
        "observation_window_days": args.window_days,
        "text_rows_available": sum(bool(row["issue_text_at_cutoff"]) for row in rows),
        "rows_with_early_event_sequence": sum(bool(row["early_event_sequence"]) for row in rows),
        "summary_snapshot_status_counts": dict(Counter(row["summary_snapshot_status"] for row in rows)),
        "description_snapshot_status_counts": dict(Counter(row["description_snapshot_status"] for row in rows)),
        "priority_snapshot_status_counts": dict(Counter(row["priority_snapshot_status"] for row in rows)),
        "issue_type_snapshot_status_counts": dict(Counter(row["issue_type_snapshot_status"] for row in rows)),
        "status_snapshot_status_counts": dict(Counter(row["status_snapshot_status"] for row in rows)),
        "resolution_delay_threshold_days_by_source": thresholds,
        "fallback_global_resolution_delay_threshold_days": global_threshold,
        "methodology_note": "Text, comments and event sequences are limited to the first observation window. Future-event columns are audit/label fields and must never be model inputs.",
    }
    summary_name = Path(args.output_name).stem + "_summary.json"
    (args.output_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
