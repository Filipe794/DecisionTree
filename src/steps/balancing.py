"""Etapa 6 (balancing). Sempre a última antes do modelo; só atua no fit."""
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.config import SEED


def smote(k_neighbors=5):
    return SMOTE(k_neighbors=k_neighbors, random_state=SEED)


def borderline_smote(k_neighbors=5):
    # Gera sintéticos só perto da fronteira entre as classes
    return BorderlineSMOTE(k_neighbors=k_neighbors, random_state=SEED)


def adasyn(n_neighbors=5):
    # Gera mais sintéticos onde a classe minoritária é mais difícil
    return ADASYN(n_neighbors=n_neighbors, random_state=SEED)


def smote_tomek():
    # SMOTE + remove pares de classes opostas coladas (limpa a fronteira)
    return SMOTETomek(random_state=SEED)


def smote_enn():
    # SMOTE + remove amostras cujos vizinhos discordam (limpeza mais agressiva)
    return SMOTEENN(random_state=SEED)


def random_over():
    return RandomOverSampler(random_state=SEED)


def random_under():
    return RandomUnderSampler(random_state=SEED)
