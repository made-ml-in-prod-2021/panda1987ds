from typing import Tuple
import logging.config
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.load_params import SplittingParams

logging.config.fileConfig('../configs/logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    logger.debug(f'Start read data from file {path}')
    try:
        data = pd.read_csv(path)
        logger.debug(f'Successful read data from file {path}')
        return data
    except FileNotFoundError:
        logger.error(f'Data file not found {path}')



def split_train_test_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, test_data
