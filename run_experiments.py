"""Executa os experimentos de um YAML.

Uso:
  python run_experiments.py experiments/experiments.yaml
  python run_experiments.py experiments/experiments.yaml --only E02_knn E04_smote
  python run_experiments.py experiments/experiments.yaml --force
"""
import argparse
import copy
import itertools
import sys
from pathlib import Path

import yaml

from src.builder import ConfigError, build_pipeline, resolve
from src.runner import run


def expand_sweep(name, cfg, sweep):
    """{model.ccp_alpha: [0, 0.01]} -> um experimento por combinação: E10__ccp_alpha=0.01."""
    if not sweep:
        return [(name, cfg)]
    keys = list(sweep)
    for key in keys:
        step, sep, param = key.partition(".")
        if not sep or step not in cfg:
            raise ConfigError(f"sweep '{key}' inválido: use etapa.parametro (ex.: model.max_depth)")
        if not isinstance(sweep[key], list):
            raise ConfigError(f"sweep '{key}' precisa de uma lista de valores")
    out = []
    for values in itertools.product(*(sweep[k] for k in keys)):
        c = copy.deepcopy(cfg)
        for key, value in zip(keys, values):
            step, _, param = key.partition(".")
            c[step][param] = value
        out.append((name + "__" + "_".join(f"{k.split('.')[1]}={v}" for k, v in zip(keys, values)), c))
    return out


def load_experiments(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    baseline = spec["baseline"]
    # O nome é só rótulo (não entra no hash): cada rodada pode ter o seu (R1_baseline...)
    baseline_name = spec.get("baseline_name", "E01_baseline")
    exps = [{"name": baseline_name, "override": {}}] + list(spec.get("experiments") or [])
    resolved = []
    for e in exps:
        try:
            cfg = resolve(baseline, e.get("override"))
            resolved += expand_sweep(e["name"], cfg, e.get("sweep"))
        except ConfigError as err:
            raise ConfigError(f"[{e['name']}] {err}") from err
    names = [n for n, _ in resolved]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ConfigError(f"Nomes repetidos no YAML: {', '.join(sorted(dup))}")
    return resolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--only", nargs="+", help="roda só estes nomes (o nome de um sweep roda todos os seus valores)")
    ap.add_argument("--force", action="store_true", help="roda de novo mesmo se já executado")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    try:
        exps = load_experiments(args.yaml)
        if args.only:
            def chosen(n, wanted):
                return n == wanted or n.startswith(wanted + "__")
            unknown = [w for w in args.only if not any(chosen(n, w) for n, _ in exps)]
            if unknown:
                raise ConfigError(f"--only com nomes inexistentes: {', '.join(sorted(unknown))}")
            exps = [(n, c) for n, c in exps if any(chosen(n, w) for w in args.only)]
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
