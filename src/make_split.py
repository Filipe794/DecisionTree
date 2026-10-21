"""Gera data/split/train.csv e test.csv (uma única vez).

Uso: python -m src.make_split [--force]
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_CSV, TRAIN_CSV, TEST_CSV, COLS, SEED, TEST_SIZE, TARGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="regera o split (invalida a comparação com execuções antigas)")
    args = ap.parse_args()

    if TRAIN_CSV.exists() and not args.force:
        print(f"Split já existe em {TRAIN_CSV.parent}. Nada feito (use --force para regerar).")
        return

    df = pd.read_csv(RAW_CSV, header=None, names=COLS)
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df[TARGET])
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"train: {len(train)} linhas | test: {len(test)} linhas")


if __name__ == "__main__":
    main()
