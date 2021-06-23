import os
import click
import logging
import sys
from shutil import copyfile

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


@click.command("preprocess")
@click.option("--input_dir")
@click.option("--output_dir")
def preprocess(input_dir: str, output_dir: str):
    logger.info(f'Start preprocess Input = {input_dir}, Output = {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    copy(input_dir, output_dir, "data.csv")
    copy(input_dir, output_dir, "target.csv")


def copy(input_dir: str, output_dir: str, filename: str):

    input = os.path.join(input_dir, filename)
    output = os.path.join(output_dir, filename)
    if not os.path.exists(input):
        logger.exception(f'File not found {input}')
        assert True
    copyfile(input, output)
    if os.path.exists(output):
        logger.info(f'File copy {output}')
    else:
        logger.error(f'Can not file copy {output}')
        assert True


if __name__ == '__main__':
    preprocess()
