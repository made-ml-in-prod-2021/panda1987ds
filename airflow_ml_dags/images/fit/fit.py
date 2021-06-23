import os
import click
import logging
import sys
import pickle
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from marshmallow import ValidationError
from typing import List, Optional
import pandas as pd
import numpy as np
from typing import Union

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
SklearnClassifier = Union[LogisticRegression, RandomForestClassifier, GaussianNB]
SCL_CLASSIFIER_DICT = {'LogisticRegression': LogisticRegression, 'RandomForestClassifier': RandomForestClassifier,
                       'GaussianNB': GaussianNB}


def read_data(path: str) -> pd.DataFrame:
    logger.debug(f'Start read data from file {path}')
    data = pd.read_csv(path)
    logger.debug(f'Successful read data from file {path}')
    return data


def dump_model(model: SklearnClassifier, transformer: ColumnTransformer, output: str):
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, 'model.pkl'), "wb") as f:
        pickle.dump(model, f)
    logger.info(f'Successful model dump {os.path.join(output, "model.pkl")}')
    with open(os.path.join(output, 'transformer.pkl'), "wb") as f:
        pickle.dump(transformer, f)
    logger.info(f'Successful model dump {os.path.join(output, "transformer.pkl")}')


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
         ("scaler", StandardScaler())
         ]
    )
    return num_pipeline


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    sq_features: Optional[List[str]]


def read_feature_params(path: str) -> FeatureParams:
    FeatureParamsSchema = class_schema(FeatureParams)
    try:
        with open(path, "r") as input_stream:
            schema = FeatureParamsSchema()
            return schema.load(yaml.safe_load(input_stream))
    except FileNotFoundError:
        logger.error(f"Can't load training parameters. File not found:{path}")
    except ValidationError as err:
        logger.error(f"Can't load training parameters. {err}")


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    logger.debug('Start build transform')
    transformer_list = [
        (
            "categorical_pipeline",
            build_categorical_pipeline(),
            params.categorical_features,
        ),
        (
            "numerical_pipeline",
            build_numerical_pipeline(),
            params.numerical_features,
        )
    ]
    transformer = ColumnTransformer(transformer_list)
    logger.debug('End build transform')
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target.values.ravel()


@click.command("fit")
@click.option("--data_path")
@click.option("--model_type")
@click.option("--output_model_path")
def fit(data_path: str, model_type: str, output_model_path: str):
    train_data = read_data(os.path.join(data_path, 'train.csv'))
    model_class = SCL_CLASSIFIER_DICT.get(model_type, None)
    if model_class is None:
        logger.error(f'Can not use this type model {model_type}')
        raise NotImplementedError()
    model = model_class()
    feature_params = read_feature_params('train_config.yaml')
    transformer = build_transformer(feature_params)
    train_features = pd.DataFrame(transformer.fit_transform(train_data))
    train_target = extract_target(train_data, feature_params)
    model.fit(train_features, train_target)
    dump_model(model, transformer, output_model_path)


if __name__ == '__main__':
    fit()
