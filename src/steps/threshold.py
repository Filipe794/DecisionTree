"""Etapa opcional (threshold): envolve o pipeline inteiro e escolhe o limiar de decisão.

A árvore prevê 1 quando a probabilidade >= 0.5. Aqui o limiar é escolhido por um CV interno,
feito só com o treino recebido, para maximizar `scoring`. Não altera o roc_auc.
"""
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
from src.config import SEED


def tuned(pipe, scoring="f1", n_splits=5):
    # Evite scoring="recall": o limiar ótimo seria 0 (tudo positivo)
    return TunedThresholdClassifierCV(
        pipe, scoring=scoring,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED))
