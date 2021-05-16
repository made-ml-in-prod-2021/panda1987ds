import logging.config
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import requests

from app import IP, PORT

os.chdir(os.path.abspath(os.path.dirname(__file__)))
logging.config.fileConfig('../configs/logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
DEFAULT_PATH_TO_DATA = '../data/request_example.json'


def setup_parser():
    parser = ArgumentParser(
        prog="request predict",
        description="tools request",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data",
        help="path to request data",
        dest="data_path",
        default=DEFAULT_PATH_TO_DATA,
    )
    return parser


def main():
    parser = setup_parser()
    arguments = parser.parse_args()
    data_path = arguments.data_path
    logger.debug(f'Read request file: {data_path}')
    try:
        with open(data_path, 'rb') as f:
            requests_json = f.read()
    except FileNotFoundError:
        logger.error(f'File not found {data_path}')

    url = f'http://{IP}:{PORT}/predict/'
    logger.debug(f'Make request to: {url}')
    response = requests.post(url, data=requests_json)

    logger.info(f'Response status code: {response.status_code}')
    if response.status_code == 200:
        logger.info(f'Response text: {response.json()}')


if __name__ == "__main__":
    main()
