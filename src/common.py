import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

BASE = Path('/content/drive/MyDrive/Mestrado/Aprendizado de Máquina/DecisionTree')
SEED = 42
COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
ZERO_COLS = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

# Mesmos folds para todo mundo => comparação justa
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
SCORING = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

def load_train():
    df = pd.read_csv(BASE/'data/split/train.csv')
    return df.drop('Outcome', axis=1), df['Outcome']

def load_test():   # usar SOMENTE no notebook 99
    df = pd.read_csv(BASE/'data/split/test.csv')
    return df.drop('Outcome', axis=1), df['Outcome']

class ZeroToNan(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X[ZERO_COLS] = X[ZERO_COLS].replace(0, np.nan)
        return X

def run_experiment(nome, pipeline, X, y, responsavel, obs=''):
    res = cross_validate(pipeline, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    linha = {'data': datetime.now().strftime('%Y-%m-%d %H:%M'),
             'experimento': nome, 'responsavel': responsavel, 'obs': obs}
    for m in SCORING:
        linha[f'{m}_media'] = res[f'test_{m}'].mean()
        linha[f'{m}_std'] = res[f'test_{m}'].std()
    arq = BASE/f'results/results_{responsavel}.csv'
    arq.parent.mkdir(exist_ok=True)
    pd.concat([pd.read_csv(arq) if arq.exists() else pd.DataFrame(),
               pd.DataFrame([linha])]).to_csv(arq, index=False)
    return linha