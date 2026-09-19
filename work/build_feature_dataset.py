import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles


DEFAULT_INPUT_FILENAMES = [
    ("SecondLife", "secondlife_5000.json"),
    ("JiraEcosystem", "jiraecosystem_5000.json"),
    ("IntelDAOS", "inteldaos_5000.json"),
    ("ApacheOldSample", "apache_500.ndjson"),
    ("ApacheRecentSample", "apache_recent_500.ndjson"),
]

DONE_STATUSES = {"done", "resolved", "closed", "complete", "completed", "fixed"}
REOPEN_STATUSES = {"open", "reopened", "reopen", "in progress", "todo", "to do"}
REQUIREMENT_FIELDS = {
    "description",
    "summary",
    "priority",
    "components",
    "component",
    "labels",
    "versions",
    "fix versions",
    "fixversions",
}


def load_json_lines(path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date(value):
    if not value:
        return None
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"]
    if not isinstance(value, str):
        return None
    value = value.replace("Z", "+0000")
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%a %b %d %H:%M:%S %Z %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    return None


def get_name(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("value") or value.get("key") or ""
    if value is None:
        return ""
    return str(value)


def as_list(value):
    return value if isinstance(value, list) else []


def text_len(value):
    return len(value) if isinstance(value, str) else 0


def collect_changelog_metrics(doc):
    histories = as_list(doc.get("changelog", {}).get("histories"))
    metrics = {
        "changelog_history_count": len(histories),
        "changelog_item_count": 0,
        "status_change_count": 0,
        "assignee_change_count": 0,
        "priority_change_count": 0,
        "description_change_count": 0,
        "summary_change_count": 0,
        "component_change_count": 0,
        "label_change_count": 0,
        "resolution_change_count": 0,
        "workflow_change_count": 0,
        "requirement_change_count": 0,
        "developer_activity_count": 0,
        "reopen_transition_count": 0,
    }
    for history in histories:
        for item in as_list(history.get("items")):
            field = str(item.get("field") or "").strip().lower()
            from_status = str(item.get("fromString") or "").strip().lower()
            to_status = str(item.get("toString") or "").strip().lower()
            metrics["changelog_item_count"] += 1

            if field == "status":
                metrics["status_change_count"] += 1
                if from_status in DONE_STATUSES and to_status in REOPEN_STATUSES:
                    metrics["reopen_transition_count"] += 1
            elif field == "assignee":
                metrics["assignee_change_count"] += 1
            elif field == "priority":
                metrics["priority_change_count"] += 1
            elif field == "description":
                metrics["description_change_count"] += 1
            elif field == "summary":
                metrics["summary_change_count"] += 1
            elif field in {"component", "components"}:
                metrics["component_change_count"] += 1
            elif field == "labels":
                metrics["label_change_count"] += 1
            elif field == "resolution":
                metrics["resolution_change_count"] += 1
            elif field == "workflow":
                metrics["workflow_change_count"] += 1

            if field in REQUIREMENT_FIELDS:
                metrics["requirement_change_count"] += 1
            if field in {"assignee", "status", "remoteissuelink", "link"}:
                metrics["developer_activity_count"] += 1
    return metrics


def flatten_issue(source, doc, now):
    fields = doc.get("fields", {})
    comments = fields.get("comments")
    if comments is None and isinstance(fields.get("comment"), dict):
        comments = fields["comment"].get("comments")
    comments = as_list(comments)

    created = parse_date(fields.get("created"))
    updated = parse_date(fields.get("updated"))
    resolved = parse_date(fields.get("resolutiondate"))

    age_days = (now - created).days if created else None
    update_age_days = (now - updated).days if updated else None
    resolution_days = (resolved - created).days if created and resolved else None

    comment_lengths = [text_len(comment.get("body")) for comment in comments if isinstance(comment, dict)]
    changelog_metrics = collect_changelog_metrics(doc)

    row = {
        "source_collection": source,
        "issue_key": doc.get("key", ""),
        "project_key": get_name(fields.get("project", {})),
        "issue_type": get_name(fields.get("issuetype")),
        "priority": get_name(fields.get("priority")),
        "status": get_name(fields.get("status")),
        "status_category": get_name(fields.get("status", {}).get("statusCategory")) if isinstance(fields.get("status"), dict) else "",
        "has_assignee": int(bool(fields.get("assignee"))),
        "components_count": len(as_list(fields.get("components"))),
        "labels_count": len(as_list(fields.get("labels"))),
        "versions_count": len(as_list(fields.get("versions"))),
        "fix_versions_count": len(as_list(fields.get("fixVersions"))),
        "issue_links_count": len(as_list(fields.get("issuelinks"))),
        "subtasks_count": len(as_list(fields.get("subtasks"))),
        "description_length": text_len(fields.get("description")),
        "summary_length": text_len(fields.get("summary")),
        "comment_count": len(comments),
        "avg_comment_length": round(sum(comment_lengths) / len(comment_lengths), 2) if comment_lengths else 0,
        "issue_age_days": age_days if age_days is not None else "",
        "days_since_last_update": update_age_days if update_age_days is not None else "",
        "resolution_days": resolution_days if resolution_days is not None else "",
        "is_resolved": int(resolved is not None),
    }
    row.update(changelog_metrics)
    row["requirement_volatility_label"] = int(row["requirement_change_count"] >= 2)
    row["issue_reopen_label"] = int(row["reopen_transition_count"] > 0)
    return row


def add_resolution_risk_labels(rows):
    resolved_days = [int(row["resolution_days"]) for row in rows if row["resolution_days"] != "" and int(row["resolution_days"]) >= 0]
    if len(resolved_days) >= 4:
        delay_threshold = quantiles(resolved_days, n=4)[2]
    elif resolved_days:
        delay_threshold = max(resolved_days)
    else:
        delay_threshold = 0

    for row in rows:
        if row["resolution_days"] != "":
            row["issue_resolution_risk_label"] = int(int(row["resolution_days"]) > delay_threshold)
        else:
            row["issue_resolution_risk_label"] = int(row["status_category"].lower() != "done" and row["days_since_last_update"] != "" and int(row["days_since_last_update"]) > 30)
    return delay_threshold


def default_workspace_root():
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return Path.cwd()


def build_paths(input_dir, output_dir):
    inputs = [(source, input_dir / filename) for source, filename in DEFAULT_INPUT_FILENAMES]
    output = output_dir / "jira_features_sample.csv"
    summary_output = output_dir / "jira_features_summary.json"
    return inputs, output, summary_output


def main():
    parser = argparse.ArgumentParser(description="Build a flat Jira ML feature dataset.")
    workspace_root = default_workspace_root()
    parser.add_argument("--input-dir", type=Path, default=workspace_root / "data" / "exports")
    parser.add_argument("--apache-dir", type=Path, default=workspace_root / "data" / "samples")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "data" / "processed")
    args, _ = parser.parse_known_args()

    inputs = [
        ("SecondLife", args.input_dir / "secondlife_5000.json"),
        ("JiraEcosystem", args.input_dir / "jiraecosystem_5000.json"),
        ("IntelDAOS", args.input_dir / "inteldaos_5000.json"),
        ("ApacheOldSample", args.apache_dir / "apache_500.ndjson"),
        ("ApacheRecentSample", args.apache_dir / "apache_recent_500.ndjson"),
    ]
    output = args.output_dir / "jira_features_sample.csv"
    summary_output = args.output_dir / "jira_features_summary.json"

    now = datetime.now(timezone.utc)
    rows = []
    source_counts = {}
    for source, path in inputs:
        count = 0
        for doc in load_json_lines(path):
            rows.append(flatten_issue(source, doc, now))
            count += 1
        source_counts[source] = count

    if not rows:
        missing = [str(path) for _, path in inputs if not path.exists()]
        raise FileNotFoundError(
            "No input records were found. Check that these files are uploaded/present: "
            + ", ".join(missing)
        )

    delay_threshold = add_resolution_risk_labels(rows)
    fieldnames = list(rows[0].keys()) if rows else []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = {
        "requirement_volatility_label": sum(row["requirement_volatility_label"] for row in rows),
        "issue_resolution_risk_label": sum(row["issue_resolution_risk_label"] for row in rows),
        "issue_reopen_label": sum(row["issue_reopen_label"] for row in rows),
    }
    summary = {
        "row_count": len(rows),
        "source_counts": source_counts,
        "resolution_delay_threshold_days": delay_threshold,
        "label_positive_counts": label_counts,
        "output_csv": str(output),
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
