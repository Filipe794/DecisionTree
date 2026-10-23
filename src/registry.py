"""Catálogo: nome usado no YAML -> função que constrói o passo.

Método novo = função no arquivo da etapa + uma linha aqui.
"""
from src.steps import balancing, features, imputers, selection, threshold, zeros

# Ordem fixa do pipeline. NÃO alterar por experimento.
# Sem etapa de outliers: árvores dividem por limiares, então valores extremos não as distorcem.
STEP_ORDER = ["zeros", "missing_flags", "imputer",
              "features", "selection", "balancing"]  # + "model" no fim
# Etapas que podem ser omitidas no YAML (equivalem a none). Envolvem o pipeline inteiro.
OPTIONAL_STEPS = ["threshold"]

REGISTRY = {
    "zeros": {
        "none": None,
        "to_nan": zeros.to_nan,
    },
    "missing_flags": {
        "none": None,
        "flags": zeros.flags,
    },
    "imputer": {
        "none": None,
        "median": imputers.median,
        "mean": imputers.mean,
        "knn": imputers.knn,
        "knn5": lambda: imputers.knn(5),
        "knn_scaled": imputers.knn_scaled,
        "iterative": imputers.iterative,
    },
    "features": {
        "none": None,
        "bmi_age_bins": features.bmi_age_bins,
        "ratios": features.ratios,
        "homa": features.homa,
    },
    "selection": {
        "none": None,
        "kbest": selection.kbest,
        "kbest8": lambda: selection.kbest(8),
        "rfe": selection.rfe,
        "rfe6": lambda: selection.rfe(6),
    },
    "balancing": {
        "none": None,
        "smote": balancing.smote,
        "random_over": balancing.random_over,
        "random_under": balancing.random_under,
        "borderline_smote": balancing.borderline_smote,
        "adasyn": balancing.adasyn,
        "smote_tomek": balancing.smote_tomek,
        "smote_enn": balancing.smote_enn,
    },
    "threshold": {
        "none": None,
        "tuned": threshold.tuned,
    },
}
