import pandas as pd
import os
import click
import logging
import sys
import pickle
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
METRICS_DICT = {'roc_auc': roc_auc_score, 'f1': f1_score, 'accuracy': accuracy_score}


def load_model(input_path: str):
    logger.info('Start load model')
    try:
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f'Model loaded successfully {str}')
        return model
    except FileNotFoundError():
        logger.error(f"Critical error. Can't load model {str}")
        raise True


def read_data(path: str) -> pd.DataFrame:
    logger.debug(f'Start read data from file {path}')
    data = pd.read_csv(path)
    logger.debug(f'Successful read data from file {path}')
    return data


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    metrics_dict = {}
    for name_metric, metric in METRICS_DICT.items():
        metrics_dict[name_metric] = metric(target, predicts)

    return metrics_dict


@click.command("validate")
@click.option("--model_path")
@click.option("--data_path")
@click.option("--output_path")
def validate(model_path: str, data_path: str, output_path: str):
    logger.info(f'Start validate model by path {model_path}')
    model = load_model(os.path.join(model_path, 'model.pkl'))
    transformer = load_model(os.path.join(model_path, 'transformer.pkl'))
    train_data = read_data(os.path.join(data_path, 'train.csv'))
    test_features = pd.DataFrame(transformer.transform(train_data))
    predict_target = pd.DataFrame(model.predict(test_features))
    os.makedirs(output_path, exist_ok=True)
    predict_target.to_csv(os.path.join(output_path, 'predictions.csv'), index=False)
    logger.info(f'Successful save predict target in file {os.path.join(output_path, "predictions.csv")}')


if __name__ == '__main__':
    validate()
