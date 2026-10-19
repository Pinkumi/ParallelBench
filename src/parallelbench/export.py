"""Export each run with its environment attached to both formats."""
import json
from pathlib import Path
import pandas as pd


def ensure_output_available(directory):
    for name in ("results.json", "results.csv", "trial_results.csv", "comparison.png"):
        path = Path(directory) / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a new output directory")


def export_results(report, directory):
    directory = Path(directory)
    ensure_output_available(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = directory / "results.json", directory / "results.csv"
    with json_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(report, indent=2, allow_nan=False))
    metadata = {key: json.dumps(value) for key, value in report.items()
                if key not in ("results", "trial_results")}
    for name, rows in (("results.csv", report["results"]),
                       ("trial_results.csv", report.get("trial_results", []))):
        records = [{**metadata, **{key: json.dumps(value) if isinstance(value, list) else value
                                  for key, value in row.items()}} for row in rows]
        if records:
            pd.DataFrame(records).to_csv(directory / name, index=False, mode="x")
    return csv_path, json_path
