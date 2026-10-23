"""Compara duas execuções fold a fold com o teste t corrigido de Nadeau e Bengio (2003).

Uso: python compare.py <pasta_run_A> <pasta_run_B> [--metric roc_auc]

As notas de k-fold repetido não são independentes (os folds compartilham dados de treino),
então o teste t comum subestima a variância e acha "diferenças" demais. A correção soma
n_val/n_tr à variância da média das diferenças.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config import N_REPEATS, N_SPLITS, RUNS_DIR, SCORING


def load_run(arg):
    """Devolve (nome da pasta, sha256 do train.csv, notas por fold)."""
    path = Path(arg)
    run_dir = path if path.is_dir() else RUNS_DIR / arg
    csv, meta = run_dir / "fold_scores.csv", run_dir / "meta.json"
    if not csv.exists():
        sys.exit(f"Execução '{arg}' não encontrada (procurei {csv})")
    if not meta.exists():
        sys.exit(f"Execução '{run_dir.name}' sem meta.json: não dá para conferir o split")
    train_sha = json.loads(meta.read_text(encoding="utf-8")).get("train_sha256")
    if not train_sha:
        sys.exit(f"O meta.json de '{run_dir.name}' não tem o campo train_sha256")
    folds = pd.read_csv(csv).sort_values(["repeat", "fold"]).reset_index(drop=True)
    return run_dir.name, train_sha, folds


def corrected_ttest(d, k=N_SPLITS, r=N_REPEATS):
    """t = mean(d) / sqrt((1/(k*r) + n_val/n_tr) * var(d)), com n_val/n_tr = 1/(k-1)."""
    var = np.var(d, ddof=1)
    if var == 0:  # diferença constante em todos os folds
        return (0.0, 1.0) if np.mean(d) == 0 else (np.inf * np.sign(np.mean(d)), 0.0)
    t = np.mean(d) / np.sqrt((1 / (k * r) + 1 / (k - 1)) * var)
    p = 2 * stats.t.sf(abs(t), df=k * r - 1)
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", help="pasta em results/runs/ (ou caminho)")
    ap.add_argument("run_b", help="pasta em results/runs/ (ou caminho)")
    ap.add_argument("--metric", choices=SCORING, help="só esta métrica (padrão: todas)")
    args = ap.parse_args()

    name_a, sha_a, a = load_run(args.run_a)
    name_b, sha_b, b = load_run(args.run_b)
    # 6 caracteres: mesmo formato da coluna "split" do summary.csv
    if sha_a != sha_b:
        sys.exit(f"As execuções usam splits diferentes (A: {sha_a[:6]}, B: {sha_b[:6]}). "
                 "Só compare execuções feitas com o mesmo train.csv.")
    n = N_SPLITS * N_REPEATS
    keys_a, keys_b = a[["repeat", "fold"]], b[["repeat", "fold"]]
    if len(a) != n or len(b) != n or not keys_a.equals(keys_b):
        sys.exit(f"As execuções não têm as mesmas {n} linhas (repeat, fold): "
                 f"A tem {len(a)}, B tem {len(b)}. Só compare execuções do mesmo CV.")

    rows = []
    for m in [args.metric] if args.metric else SCORING:
        d = (b[m] - a[m]).to_numpy()
        t, p = corrected_ttest(d)
        rows.append({"métrica": m, "média A": a[m].mean(), "média B": b[m].mean(),
                     "dif (B-A)": d.mean(), "t": t, "p": p})

    print(f"A: {name_a}\nB: {name_b}\nsplit: {sha_a[:6]}")
    print(f"Teste t corrigido (Nadeau e Bengio), {N_SPLITS} folds x {N_REPEATS} repetições, "
          f"{n - 1} graus de liberdade\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
