"""Junta todas as execuções de results/runs/ numa tabela (sempre regenerada).

Uso: python summarize.py [--metric roc_auc]
"""
import argparse
import json

import pandas as pd
import yaml

from src.config import PRIMARY_METRIC, RUNS_DIR, SCORING, SUMMARY_CSV
from src.registry import STEP_ORDER


def describe(cfg):
    """Resumo curto: só as etapas diferentes de none, ex.: 'to_nan > knn(n_neighbors=5) | max_depth=4'."""
    parts = []
    for step in STEP_ORDER:
        params = dict(cfg[step])
        name = params.pop("name")
        if name != "none":
            args = ",".join(f"{k}={v}" for k, v in params.items())
            parts.append(f"{name}({args})" if args else name)
    model = ",".join(f"{k}={v}" for k, v in (cfg.get("model") or {}).items())
    return " > ".join(parts) + f" | {model}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=PRIMARY_METRIC, choices=SCORING)
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*/")):
        if run_dir.name.startswith(".tmp_") or not (run_dir / "meta.json").exists():
            continue
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        row = {"run_id": meta["run_id"], "name": meta["name"], "date": meta["date"]}
        for m in SCORING:
            row[f"{m}_mean"] = metrics[m]["mean"]
            row[f"{m}_std"] = metrics[m]["std"]
        row["pipeline"] = describe(cfg)
        rows.append(row)

    if not rows:
        print("Nenhuma execução em results/runs/.")
        return

    df = pd.DataFrame(rows).sort_values(f"{args.metric}_mean", ascending=False)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)

    show = ["name"] + [f"{m}_mean" for m in SCORING] + [f"{args.metric}_std", "pipeline"]
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(df[show].to_string(index=False, float_format="%.3f"))
    print(f"\n{len(df)} execuções -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
