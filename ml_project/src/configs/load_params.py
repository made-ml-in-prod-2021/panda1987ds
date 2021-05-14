from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from marshmallow import ValidationError
import logging.config
import yaml
from typing import List, Optional

logging.config.fileConfig('../configs/logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    sq_features: Optional[List[str]]


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    model_param: dict = field(default=None)


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    try:
        with open(path, "r") as input_stream:
            schema = TrainingPipelineParamsSchema()
            return schema.load(yaml.safe_load(input_stream))
    except FileNotFoundError:
        logger.error(f"Can't load training parameters. File not found:{path}")
    except ValidationError as err:
        logger.error(f"Can't load training parameters. {err}")
