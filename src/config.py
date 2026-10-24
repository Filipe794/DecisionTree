from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

# Raiz do projeto calculada a partir deste arquivo: funciona local e no Colab
BASE = Path(__file__).resolve().parents[1]
RAW_CSV = BASE / "data/raw/pima-indians-diabetes.csv"
TRAIN_CSV = BASE / "data/split/train.csv"
TEST_CSV = BASE / "data/split/test.csv"
RUNS_DIR = BASE / "results/runs"
FINAL_DIR = BASE / "results/final"
SUMMARY_CSV = BASE / "results/summary.csv"

SEED = 42
TEST_SIZE = 0.3

COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
TARGET = "Outcome"
FEATURES = [c for c in COLS if c != TARGET]
# Colunas em que 0 é fisiologicamente impossível => valor ausente
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Mesmos folds para todo mundo => comparação justa
N_SPLITS, N_REPEATS = 5, 5
CV = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
# balanced_accuracy = média do recall das duas classes (critério do R3, junto com o F1)
SCORING = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"]
PRIMARY_METRIC = "roc_auc"

# Nível de significância do teste t corrigido (bilateral) usado pelo compare.py.
# Não entra no hash das execuções: mudar o ALPHA só muda a leitura das comparações.
ALPHA = 0.05
