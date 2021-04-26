from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

from configs.load_params import FeatureParams


class SquaringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X, y=None):
        that = X * X
        return that


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


def build_sq_pipeline() -> Pipeline:
    sq_pipeline = Pipeline(
        [("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
         ("squaring", SquaringTransformer()),
         ("scaler", StandardScaler())
         ]
    )
    return sq_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame, transform_only: bool = False) -> pd.DataFrame:
    if transform_only:
        return pd.DataFrame(transformer.transform(df))
    else:
        return pd.DataFrame(transformer.fit_transform(df))


def build_transformer(params: FeatureParams) -> ColumnTransformer:
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
    if params.sq_features is not None:
        transformer_list.append(
            (
                "sq_pipeline",
                build_sq_pipeline(),
                params.sq_features,
            ))

    transformer = ColumnTransformer(transformer_list)
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target.values.ravel()
