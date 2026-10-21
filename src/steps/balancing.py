"""Etapa 7 (balancing). Sempre a última antes do modelo; só atua no fit."""
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.config import SEED


def smote(k_neighbors=5):
    return SMOTE(k_neighbors=k_neighbors, random_state=SEED)


def random_over():
    return RandomOverSampler(random_state=SEED)


def random_under():
    return RandomUnderSampler(random_state=SEED)
