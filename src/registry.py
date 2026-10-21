"""Catálogo: nome usado no YAML -> função que constrói o passo.

Método novo = função no arquivo da etapa + uma linha aqui.
"""
from src.steps import balancing, features, imputers, outliers, selection, zeros

# Ordem fixa do pipeline. NÃO alterar por experimento.
STEP_ORDER = ["zeros", "missing_flags", "outliers", "imputer",
              "features", "selection", "balancing"]  # + "model" no fim

REGISTRY = {
    "zeros": {
        "none": None,
        "to_nan": zeros.to_nan,
    },
    "missing_flags": {
        "none": None,
        "flags": zeros.flags,
    },
    "outliers": {
        "none": None,
        "winsor_1_99": lambda: outliers.winsor(0.01, 0.99),
        "winsor_5_95": lambda: outliers.winsor(0.05, 0.95),
        "winsor": outliers.winsor,
        "iqr_clip": outliers.iqr_clip,
        "iqr_drop": outliers.iqr_drop,
    },
    "imputer": {
        "none": None,
        "median": imputers.median,
        "mean": imputers.mean,
        "knn": imputers.knn,
        "knn5": lambda: imputers.knn(5),
        "iterative": imputers.iterative,
    },
    "features": {
        "none": None,
        "bmi_age_bins": features.bmi_age_bins,
        "ratios": features.ratios,
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
    },
}
