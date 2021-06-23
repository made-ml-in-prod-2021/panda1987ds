from numpy.random import randint, normal
import pandas as pd
import os
import click
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def generate_categorical_features(size_df) -> dict:
    return {
        'cp': randint(0, 4, size_df),
        'sex': randint(0, 2, size_df),
        'fbs': randint(0, 2, size_df),
        'restecg': randint(0, 2, size_df),
        'exang': randint(0, 2, size_df),
        'slope': randint(0, 3, size_df),
        'ca': randint(0, 5, size_df),
        'thal': randint(1, 4, size_df)
    }


def generate_numeric_features(size_df) -> dict:
    oldpeak = normal(0, 2, size_df).round(2)
    oldpeak[oldpeak < 0] = 0
    return {
        'age': normal(54, 9, size_df).astype(int),
        'trestbps': normal(132, 17, size_df).astype(int),
        'chol': normal(246, 52, size_df).astype(int),
        'thalach': normal(150, 23, size_df).astype(int),
        'oldpeak': oldpeak
    }


def generate_target_dict(size_df) -> dict:
    return {'target': randint(0, 2, size_df)}


@click.command("generate")
@click.option("--path")
def generate_date(path: str) -> pd.DataFrame:
    logger.info(f'Start generate by path {path}')
    size_df = randint(100, 200)
    categorical_features = generate_categorical_features(size_df)
    numeric_features = generate_numeric_features(size_df)
    target = pd.DataFrame(generate_target_dict(size_df))
    date = pd.DataFrame({**categorical_features, **numeric_features})
    os.makedirs(path, exist_ok=True)
    try:
        date.to_csv(os.path.join(path, "data.csv"), index=False)
        logger.info('File "data.csv" saved')
        target.to_csv(os.path.join(path, "target.csv"), index=False)
        logger.info('File "target.csv" saved')
    except PermissionError as ex:
        logger.error('Can not save file. Permission error. {ex}')


if __name__ == '__main__':
    generate_date()
