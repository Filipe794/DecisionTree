"""Avaliação final no conjunto de teste (rodar só no fim, uma vez por execução escolhida).

Uso: python final_test.py --run 20261021_143012_a3f9c1_E05_knn_smote
"""
import argparse
import json
import sys
from datetime import datetime

import joblib
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, get_scorer

from src.builder import build_pipeline
from src.config import FINAL_DIR, RUNS_DIR, SCORING
from src.data import load_test, load_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nome da pasta em results/runs/")
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"Execução '{args.run}' não encontrada em {RUNS_DIR}")
    out_dir = FINAL_DIR / args.run
    if out_dir.exists():
        print(f"Já avaliado no teste. Resultado guardado em {out_dir}:")
        print((out_dir / "metrics.json").read_text(encoding="utf-8"))
        return

    X_train, y_train = load_train()
    X_test, y_test = load_test()
    if (run_dir / "pipeline.joblib").exists():
        # Avalia exatamente a árvore salva na execução
        pipe = joblib.load(run_dir / "pipeline.joblib")
    else:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        cfg.pop("name", None)
        pipe = build_pipeline(cfg).fit(X_train, y_train)
    # Mesmas métricas e mesmos scorers da validação cruzada (src/config.py:SCORING)
    metrics = {m: float(get_scorer(m)(pipe, X_test, y_test)) for m in SCORING}
    cm = confusion_matrix(y_test, pipe.predict(X_test))

    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
        out_dir / "confusion_matrix.csv")
    (out_dir / "meta.json").write_text(json.dumps(
        {"run": args.run, "date": datetime.now().isoformat(timespec="seconds"),
         "n_train": len(X_train), "n_test": len(X_test)}, indent=2), encoding="utf-8")

    for m, v in metrics.items():
        print(f"{m:>17}: {v:.3f}")
    print(f"Matriz de confusão:\n{cm}\n-> {out_dir}")


if __name__ == "__main__":
    main()
