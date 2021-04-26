import pandas as pd

import data.load_split_data as load_split_data
from configs.load_params import TrainingPipelineParams


def test_read_data(random_df: pd.DataFrame, input_data_path_test: str):
    read_df = load_split_data.read_data(input_data_path_test)
    assert ((read_df.columns == random_df.columns).all())
    assert ((read_df == random_df).all().all())


def test_split_train_test_data(random_df: pd.DataFrame, training_pipeline_params: TrainingPipelineParams):
    train, test = load_split_data.split_train_test_data(random_df, training_pipeline_params.splitting_params)
    assert (isinstance(train, pd.DataFrame))
    assert (isinstance(test, pd.DataFrame))
    assert (round(len(test) / (len(train) + len(test)), 1) == training_pipeline_params.splitting_params.val_size)
