import sys


from typing import Dict, Union

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score

from configs.load_params import TrainingParams

SklearnClassifier = Union[LogisticRegression, RandomForestClassifier, GaussianNB]
SCL_CLASSIFIER_DICT = {'LogisticRegression': LogisticRegression, 'RandomForestClassifier': RandomForestClassifier,
                       'GaussianNB': GaussianNB}
METRICS_DICT = {'roc_auc': roc_auc_score, 'f1': f1_score, 'accuracy': accuracy_score}


def train(features: pd.DataFrame, target: pd.Series, train_params: TrainingParams) -> SklearnClassifier:
    model_class = SCL_CLASSIFIER_DICT.get(train_params.model_type, None)
    if model_class is None:
        raise NotImplementedError()

    model = model_class(**train_params.model_param) if train_params.model_param is not None else model_class()

    model.fit(features, target)
    return model


def predict(model: SklearnClassifier, features: pd.DataFrame) -> np.array:
    return model.predict(features)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    metrics_dict = {}
    for name_metric, metric in METRICS_DICT.items():
        metrics_dict[name_metric] = metric(target, predicts)

    return metrics_dict


def cross_val_model(model: SklearnClassifier, features: pd.DataFrame, target: pd.Series) -> Dict[str, list]:
    metrics_dict = {}
    for name_metric, metric in METRICS_DICT.items():
        metrics_dict[name_metric + '_cross_val'] = cross_val_score(model, features, target,
                                                                   scoring=make_scorer(metric, greater_is_better=True),
                                                                   cv=10).tolist()
    return metrics_dict


def dump_model(model: SklearnClassifier, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(input: str) -> SklearnClassifier:
    with open(input, 'rb') as f:
        model = pickle.load(f)
    return model
