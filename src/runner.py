"""Roda a validação cruzada de um experimento e grava numa pasta nova e imutável."""
import hashlib
import json
import platform
import shutil
import time
from datetime import datetime

import imblearn
import joblib
from joblib import parallel_config
import matplotlib
matplotlib.use("Agg")  # grava PNG sem abrir janela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import cross_validate
from sklearn.tree import export_text, plot_tree

from src.builder import build_pipeline, unwrap
from src.config import CV, N_REPEATS, N_SPLITS, RUNS_DIR, SCORING, SEED, TRAIN_CSV
from src.data import file_sha256, load_train


def config_hash(cfg, train_sha):
    """Identidade do experimento: config completa + seed + CV + métricas + treino."""
    payload = {"config": cfg, "seed": SEED, "cv": [N_SPLITS, N_REPEATS],
               "scoring": SCORING, "train_sha256": train_sha}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def save_tree(pipe, out_dir):
    """Grava o pipeline treinado e representações legíveis da árvore."""
    joblib.dump(pipe, out_dir / "pipeline.joblib")
    tree = unwrap(pipe).named_steps["model"]
    names = [str(c) for c in tree.feature_names_in_]
    (out_dir / "tree.txt").write_text(export_text(tree, feature_names=names, show_weights=True),
                                      encoding="utf-8")
    (pd.DataFrame({"feature": names, "importance": tree.feature_importances_})
     .sort_values("importance", ascending=False)
     .to_csv(out_dir / "feature_importances.csv", index=False))

    leaves = tree.get_n_leaves()
    fig, ax = plt.subplots(figsize=(max(12, leaves * 1.6), max(6, tree.get_depth() * 2.2)))
    plot_tree(tree, feature_names=names, class_names=["0", "1"], filled=True,
              rounded=True, fontsize=8, ax=ax)
    fig.savefig(out_dir / "tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def find_existing(full_hash):
    if not RUNS_DIR.exists():
        return None
    for meta in sorted(RUNS_DIR.glob("*/meta.json")):
        if json.loads(meta.read_text(encoding="utf-8")).get("hash") == full_hash:
            return meta.parent
    return None


def run(name, cfg, force=False, n_jobs=-1):
    train_sha = file_sha256(TRAIN_CSV)
    full_hash = config_hash(cfg, train_sha)
    existing = find_existing(full_hash)
    if existing and not force:
        print(f"[{name}] já executado em {existing.name}, pulando (use --force para repetir)")
        return existing

    X, y = load_train()
    pipe = build_pipeline(cfg)
    t0 = time.time()
    # Threads em vez de processos: no Python 3.14 o joblib (loky) quebra quando a pasta
    # temporária tem acento ("Máquina"). Com ~500 linhas, threads custam o mesmo.
    with parallel_config(backend="threading"):
        res = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=n_jobs)
    elapsed = time.time() - t0

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{full_hash[:6]}_{name}"
    final_dir = RUNS_DIR / run_id
    tmp_dir = RUNS_DIR / f".tmp_{run_id}"
    tmp_dir.mkdir(parents=True)
    try:
        (tmp_dir / "config.yaml").write_text(
            yaml.safe_dump({"name": name, **cfg}, sort_keys=False, allow_unicode=True),
            encoding="utf-8")

        metrics = {m: {"mean": float(np.mean(res[f"test_{m}"])),
                       "std": float(np.std(res[f"test_{m}"]))} for m in SCORING}
        (tmp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        n = len(res["fit_time"])
        folds = pd.DataFrame({"repeat": np.arange(n) // N_SPLITS, "fold": np.arange(n) % N_SPLITS,
                              **{m: res[f"test_{m}"] for m in SCORING},
                              "fit_time": res["fit_time"]})
        folds.to_csv(tmp_dir / "fold_scores.csv", index=False)

        # Árvore final: mesmo pipeline treinado no treino inteiro (os 25 modelos do CV são descartados)
        final_pipe = build_pipeline(cfg).fit(X, y)
        save_tree(final_pipe, tmp_dir)

        meta = {"run_id": run_id, "name": name, "hash": full_hash,
                "date": datetime.now().isoformat(timespec="seconds"),
                "train_sha256": train_sha, "n_train": int(len(X)), "seed": SEED,
                "cv": f"RepeatedStratifiedKFold({N_SPLITS}x{N_REPEATS})",
                "elapsed_s": round(elapsed, 2),
                "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                             "imblearn": imblearn.__version__, "pandas": pd.__version__,
                             "numpy": np.__version__}}
        if hasattr(final_pipe, "best_threshold_"):
            # Limiar escolhido no treino inteiro (é o usado pela árvore salva e pelo final_test)
            meta["threshold"] = float(final_pipe.best_threshold_)
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # Só vira uma execução "válida" depois de tudo gravado
        tmp_dir.rename(final_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    summary = " | ".join(f"{m}={v['mean']:.3f}±{v['std']:.3f}" for m, v in metrics.items())
    print(f"[{name}] {summary}  ->  {run_id}")
    return final_dir
