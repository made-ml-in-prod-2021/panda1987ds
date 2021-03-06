from numpy.random import randint, normal
import pytest
import pandas as pd
import sys, os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src')))

import configs.load_params as load_params
import features.build_features as build_features
import models.fit_predict as fit_predict


@pytest.fixture()
def model_path_test(tmpdir):
    return str(tmpdir + '/model')


@pytest.fixture()
def metric_path_test(tmpdir):
    return str(tmpdir + '/metric')


@pytest.fixture()
def input_data_path_test(tmpdir, random_df):
    path = tmpdir + '/tmp_df.csv'
    random_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def params_yaml_test(metric_path_test, model_path_test, input_data_path_test):
    feature_params = {'categorical_features': ['cp', 'sex', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
                      'numerical_features': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
                      'target_col': 'target', 'sq_features': ['age']}
    splitting_params = {'val_size': 0.1, 'random_state': 42}
    training_params = {'model_type': 'LogisticRegression'}
    return {'input_data_path': input_data_path_test, 'model_path': model_path_test,
            'metric_path': metric_path_test,
            'feature_params': feature_params, 'train_params': training_params,
            'splitting_params': splitting_params}


@pytest.fixture()
def params_fio_fixture(tmpdir, params_yaml_test):
    params_fio = tmpdir.join("train_config.yaml")
    with open(params_fio, 'w') as file:
        yaml.safe_dump(params_yaml_test, file, default_flow_style=False)
    return params_fio


@pytest.fixture()
def training_pipeline_params(params_fio_fixture):
    return load_params.read_training_pipeline_params(params_fio_fixture)


@pytest.fixture()
def size_df():
    return 500


@pytest.fixture()
def categorical_features(size_df) -> dict:
    return {
        'cp': randint(0, 4, size_df),
        'sex': randint(0, 2, size_df),
        'fbs': randint(0, 2, size_df),
        'restecg': randint(0, 2, size_df),
        'exang': randint(0, 2, size_df),
        'slope': randint(0, 3, size_df),
        'ca': randint(0, 5, size_df),
        'thal': randint(1, 4, size_df)
    }


@pytest.fixture()
def numeric_features(size_df) -> dict:
    oldpeak = normal(0, 2, size_df).round(2)
    oldpeak[oldpeak < 0] = 0
    return {
        'age': normal(54, 9, size_df).astype(int),
        'trestbps': normal(132, 17, size_df).astype(int),
        'chol': normal(246, 52, size_df).astype(int),
        'thalach': normal(150, 23, size_df).astype(int),
        'oldpeak': oldpeak
    }


@pytest.fixture()
def target_dict(size_df) -> dict:
    return {'target': randint(0, 2, size_df)}


@pytest.fixture()
def sq_features(size_df) -> dict:
    return {'sq': randint(0, 5, size_df)}


@pytest.fixture()
def random_df(categorical_features, numeric_features, sq_features, target_dict) -> pd.DataFrame:
    return pd.DataFrame({**categorical_features, **numeric_features, **sq_features, **target_dict})


@pytest.fixture()
def features(random_df, training_pipeline_params):
    return build_features.make_features(build_features.build_transformer(training_pipeline_params.feature_params),
                                        random_df)


@pytest.fixture()
def target(random_df, training_pipeline_params):
    return build_features.extract_target(random_df, training_pipeline_params.feature_params)


@pytest.fixture()
def model_train(features, target, training_pipeline_params):
    return fit_predict.train(features, target, training_pipeline_params.train_params)


@pytest.fixture()
def predict_array(model_train, features):
    return fit_predict.predict(model_train, features)
