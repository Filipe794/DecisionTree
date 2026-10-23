"""config (dict) -> Pipeline do imblearn."""
import copy
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED
from src.registry import OPTIONAL_STEPS, REGISTRY, STEP_ORDER


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
        cfg[key] = copy.deepcopy(value)
    valid = STEP_ORDER + ["model"] + OPTIONAL_STEPS
    unknown = [k for k in cfg if k not in valid]
    if unknown:
        raise ConfigError(f"Etapa '{unknown[0]}' não existe. Etapas: {', '.join(valid)}")
    missing = [s for s in STEP_ORDER + ["model"] if s not in cfg]
    if missing:
        raise ConfigError(f"Config sem as etapas: {', '.join(missing)}")
    # Opcionais só entram na config se escritas, para não mudar o hash de configs antigas
    for step in STEP_ORDER + [s for s in OPTIONAL_STEPS if s in cfg]:
        cfg[step] = normalize_step(cfg[step])
    cfg["model"] = dict(cfg["model"] or {})
    return cfg


def _make(step, spec, *args):
    """Constrói o passo pelo REGISTRY; devolve None para 'none'."""
    params = dict(spec)
    name = params.pop("name")
    options = REGISTRY[step]
    if name not in options:
        raise ConfigError(f"{step} '{name}' não existe, opções: {', '.join(options)}")
    factory = options[name]
    if factory is None:
        if params:
            raise ConfigError(f"{step} 'none' não aceita parâmetros: {params}")
        return None
    try:
        return factory(*args, **params)
    except TypeError as e:
        raise ConfigError(f"{step} '{name}': parâmetros inválidos {params} ({e})") from e


def build_pipeline(cfg):
    steps = [(s, obj) for s in STEP_ORDER if (obj := _make(s, cfg[s])) is not None]
    if cfg["imputer"]["name"] == "none" and cfg["balancing"]["name"] != "none":
        # A árvore aceita NaN, mas os balanceadores do imblearn não
        raise ConfigError(f"balancing '{cfg['balancing']['name']}' exige imputer (não aceita NaN)")
    try:
        model = DecisionTreeClassifier(random_state=SEED, **cfg["model"])
    except TypeError as e:
        raise ConfigError(f"model: hiperparâmetro inválido ({e})") from e
    pipe = Pipeline(steps + [("model", model)])
    if "threshold" in cfg:
        pipe = _make("threshold", cfg["threshold"], pipe) or pipe
    return pipe


def unwrap(pipe):
    """Pipeline treinado, mesmo quando envolto pelo ajuste de limiar."""
    return getattr(pipe, "estimator_", pipe)
