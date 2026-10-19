"""Export each run with its environment attached to both formats."""
import json
from pathlib import Path
import pandas as pd


def export_results(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = directory / "results.json", directory / "results.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    metadata = {key: json.dumps(value) for key, value in report.items() if key != "results"}
    records = [{**metadata, **row, "samples_ms": json.dumps(row["samples_ms"])} for row in report["results"]]
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path, json_path
