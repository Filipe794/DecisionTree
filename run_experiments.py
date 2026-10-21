"""Executa os experimentos de um YAML.

Uso:
  python run_experiments.py experiments/experiments.yaml
  python run_experiments.py experiments/experiments.yaml --only E02_knn E04_smote
  python run_experiments.py experiments/experiments.yaml --force
"""
import argparse
import sys
from pathlib import Path

import yaml

from src.builder import ConfigError, build_pipeline, resolve
from src.runner import run


def load_experiments(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    baseline = spec["baseline"]
    exps = [{"name": "E01_baseline", "override": {}}] + list(spec.get("experiments") or [])
    names = [e["name"] for e in exps]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ConfigError(f"Nomes repetidos no YAML: {', '.join(sorted(dup))}")
    resolved = []
    for e in exps:
        try:
            resolved.append((e["name"], resolve(baseline, e.get("override"))))
        except ConfigError as err:
            raise ConfigError(f"[{e['name']}] {err}") from err
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--only", nargs="+", help="roda só estes nomes")
    ap.add_argument("--force", action="store_true", help="roda de novo mesmo se já executado")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    try:
        exps = load_experiments(args.yaml)
        if args.only:
            unknown = set(args.only) - {n for n, _ in exps}
            if unknown:
                raise ConfigError(f"--only com nomes inexistentes: {', '.join(sorted(unknown))}")
            exps = [(n, c) for n, c in exps if n in args.only]
        # Valida tudo antes de rodar qualquer coisa: erro de digitação falha na hora
        for name, cfg in exps:
            try:
                build_pipeline(cfg)
            except ConfigError as e:
                raise ConfigError(f"[{name}] {e}") from e
    except ConfigError as e:
        sys.exit(f"Erro de configuração: {e}")

    for name, cfg in exps:
        run(name, cfg, force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
