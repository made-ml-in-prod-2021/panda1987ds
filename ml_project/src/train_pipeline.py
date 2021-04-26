import logging, os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json

from configs.load_params import read_training_pipeline_params, TrainingPipelineParams
from data.load_split_data import read_data, split_train_test_data
from features.build_features import build_transformer, extract_target, make_features
from models.fit_predict import train, predict, dump_model, evaluate_model, cross_val_model, load_model

os.chdir(os.path.abspath(os.path.dirname(__file__)))
logger = logging.getLogger('train_model')
DEFAULT_PATH_TO_CONFIG = '../configs/train_config.yaml'
DEFAULT_MODE = 'train predict'
TRAIN_MODE = 'train'
PREDICT_MODE = 'predict'


def setup_logging():
    w_format = logging.Formatter('%(levelname)s: %(message)s')
    logging.basicConfig(
        filename="../logs/train_model.log", level=logging.DEBUG, format='%(levelname)s: %(message)s'
    )
    w_log = logging.FileHandler('../logs/train_model.warm')
    w_log.setLevel(logging.WARNING)
    w_log.setFormatter(w_format)
    logger.addHandler(w_log)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)


def setup_parser():
    parser = ArgumentParser(
        prog="train-model",
        description="tools to train, predict and evaluate model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        help="path to config",
        dest="config_path",
        default=DEFAULT_PATH_TO_CONFIG,
    )
    parser.add_argument(
        "-m", "--mode",
        help="select train or test",
        dest="mode",
        default=DEFAULT_MODE,
    )
    return parser


def train_pipeline(training_pipeline_params: TrainingPipelineParams, mode: str):
    logger.debug('Start pipeline')
    logger.debug('Start load data')
    all_data = read_data(training_pipeline_params.input_data_path)
    logger.debug('Data uploaded')
    transformer = build_transformer(training_pipeline_params.feature_params)
    train_data, test_data = split_train_test_data(all_data, training_pipeline_params.splitting_params)
    train_features = make_features(transformer, train_data)
    test_features = make_features(transformer, test_data, True)
    train_target = extract_target(train_data, training_pipeline_params.feature_params)
    test_target = extract_target(test_data, training_pipeline_params.feature_params)
    if mode == TRAIN_MODE or mode == DEFAULT_MODE:
        logger.debug('Start train model')
        model = train(train_features, train_target, training_pipeline_params.train_params)
        logger.debug('Training completed')
        try:
            dump_model(model, training_pipeline_params.output_model_path)
            logger.info(f'Model dump: {training_pipeline_params.output_model_path}')
        except FileNotFoundError:
            logger.warning(f'Can not dump model: {training_pipeline_params.output_model_path}')
    else:
        logger.debug('Start load model')
        try:
            model = load_model(training_pipeline_params.output_model_path)
            logger.info(f'Model uploaded with: {training_pipeline_params.output_model_path}')
        except FileNotFoundError:
            logger.warning(f'Can not load model: {training_pipeline_params.output_model_path}')
            return None, None, None
    if mode == DEFAULT_MODE or mode == PREDICT_MODE:
        predict_target = predict(model, test_features)
        metrics = evaluate_model(predict_target, test_target)
        cross_val_metric = cross_val_model(model, train_features, train_target)
        with open(training_pipeline_params.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
            json.dump(cross_val_metric, metric_file)
        logger.debug(f'Metric save at {training_pipeline_params.metric_path}')
        logger.info(f'Metrics is:')
        for metric in metrics:
            logger.info(f"{metric}: {metrics[metric]:.2f}")
        for metric in cross_val_metric:
            logger.info(f'{metric}:')
            logger.info(' '.join(f'{x:.2f}' for x in cross_val_metric[metric]))
        return model, metrics, cross_val_metric
    else:
        return model, None, None




def train_pipeline_command(config_path: str, mode: str):
    if mode != DEFAULT_MODE and mode != TRAIN_MODE and mode != PREDICT_MODE:
        logger.warning(f'Please only use the "{PREDICT_MODE}" or "{TRAIN_MODE}" mode or use program without "mode"')
        return
    logger.info(f'Read parameters from {config_path}')
    training_pipeline_params = read_training_pipeline_params(config_path)
    train_pipeline(training_pipeline_params, mode)


def main():
    setup_logging()
    parser = setup_parser()
    arguments = parser.parse_args()
    train_pipeline_command(arguments.config_path, arguments.mode)


if __name__ == "__main__":
    main()
