import os
import click
import logging
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def read_data(path: str) -> pd.DataFrame:
    logger.debug(f'Start read data from file {path}')
    data = pd.read_csv(path)
    logger.debug(f'Successful read data from file {path}')
    return data


def save_data(train: pd.DataFrame, test: pd.DataFrame, path: str):
    logger.debug(f'Start save split data in {path}')
    train.to_csv(os.path.join(path, 'train.csv'))
    logger.debug(f'Successful save train data {path}')
    test.to_csv(os.path.join(path, 'test.csv'))
    logger.debug(f'Successful save test data {path}')


@click.command("split")
@click.option("--path")
@click.option("--val_size", type=click.FloatRange(0, 1), default=0.2)
@click.option("--random_state", type=click.INT, default=42)
def split_train_test_data(path: str, val_size: float, random_state: int):
    logger.debug(f'Start split data from file {path}')
    data = read_data(os.path.join(path, 'data.csv'))
    target = read_data(os.path.join(path,'target.csv'))
    data['target'] = target['target']
    train_data, test_data = train_test_split(
        data, test_size=val_size, random_state=random_state
    )
    logger.debug(f'Successful data split')
    save_data(train_data, test_data, path)


if __name__ == '__main__':
    split_train_test_data()


