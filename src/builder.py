"""config (dict) -> Pipeline do imblearn."""
import copy
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED
from src.registry import REGISTRY, STEP_ORDER


class ConfigError(ValueError):
    pass


def normalize_step(value):
    """'knn' -> {'name': 'knn'};  {'name': 'knn', 'n_neighbors': 5} fica igual."""
    if value is None:
        return {"name": "none"}
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, dict) and "name" in value:
        return dict(value)
    raise ConfigError(f"Formato inválido: {value!r}. Use 'nome' ou {{name: nome, ...}}")


def resolve(baseline, override=None):
    """Junta baseline + override. O override SUBSTITUI a etapa inteira."""
    cfg = copy.deepcopy(baseline)
    for key, value in (override or {}).items():
        if key not in STEP_ORDER and key != "model":
            raise ConfigError(f"Etapa '{key}' não existe. Etapas: {', '.join(STEP_ORDER + ['model'])}")
        cfg[key] = copy.deepcopy(value)
    missing = [s for s in STEP_ORDER + ["model"] if s not in cfg]
    if missing:
        raise ConfigError(f"Config sem as etapas: {', '.join(missing)}")
    for step in STEP_ORDER:
        cfg[step] = normalize_step(cfg[step])
    cfg["model"] = dict(cfg["model"] or {})
    return cfg


def build_pipeline(cfg):
    steps = []
    for step in STEP_ORDER:
        params = dict(cfg[step])
        name = params.pop("name")
        options = REGISTRY[step]
        if name not in options:
            raise ConfigError(f"{step} '{name}' não existe, opções: {', '.join(options)}")
        factory = options[name]
        if factory is None:
            if params:
                raise ConfigError(f"{step} 'none' não aceita parâmetros: {params}")
            continue
        try:
            steps.append((step, factory(**params)))
        except TypeError as e:
            raise ConfigError(f"{step} '{name}': parâmetros inválidos {params} ({e})") from e
    try:
        model = DecisionTreeClassifier(random_state=SEED, **cfg["model"])
    except TypeError as e:
        raise ConfigError(f"model: hiperparâmetro inválido ({e})") from e
    steps.append(("model", model))
    return Pipeline(steps)
