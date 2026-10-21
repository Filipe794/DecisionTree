"""Etapa 6 (selection)."""
from functools import partial
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from src.config import SEED


def kbest(k=8, score="f_classif"):
    # mutual_info é aleatório: sem seed, a mesma config daria notas diferentes
    funcs = {"f_classif": f_classif,
             "mutual_info": partial(mutual_info_classif, random_state=SEED)}
    return SelectKBest(funcs[score], k=k)


def rfe(n_features=6):
    return RFE(DecisionTreeClassifier(random_state=SEED), n_features_to_select=n_features)
