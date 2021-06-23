import pandas as pd
import os
import click
import logging
import sys
import pickle
import numpy as np
from typing import Dict
import json
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
METRICS_DICT = {'roc_auc': roc_auc_score, 'f1': f1_score, 'accuracy': accuracy_score}


def load_model(input_path: str):
    logger.info('Start load model')
    try:
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
        logger.info('Model loaded successfully')
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
def validate(model_path: str, data_path: str):
    logger.info(f'Start validate model by path {model_path}')
    model = load_model(os.path.join(model_path, 'model.pkl'))
    transformer = load_model(os.path.join(model_path, 'transformer.pkl'))
    test_data = read_data(os.path.join(data_path, 'test.csv'))
    test_features = pd.DataFrame(transformer.transform(test_data))
    predict_target = model.predict(test_features)
    test_target = test_data['target'].values.ravel()
    metrics = evaluate_model(predict_target, test_target)
    with open(os.path.join(model_path, 'metric.json'), "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f'Successful save metrics in file {os.path.join(model_path, "metric.json")}')


if __name__ == '__main__':
    validate()
