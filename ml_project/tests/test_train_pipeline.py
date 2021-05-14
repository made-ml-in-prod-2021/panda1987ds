import os

from train_pipeline import train_pipeline, DEFAULT_MODE, TRAIN_MODE,PREDICT_MODE
from configs.load_params import TrainingPipelineParams


def test_train_pipeline(training_pipeline_params: TrainingPipelineParams, model_path_test: str,
                        metric_path_test: str):
    train_pipeline(training_pipeline_params, TRAIN_MODE)
    assert os.path.exists(model_path_test)
    assert not os.path.exists(metric_path_test)
    train_pipeline(training_pipeline_params, PREDICT_MODE)
    assert os.path.exists(metric_path_test)
    os.remove(model_path_test)
    os.remove(metric_path_test)
    train_pipeline(training_pipeline_params, DEFAULT_MODE)
    assert os.path.exists(model_path_test)
    assert os.path.exists(metric_path_test)
