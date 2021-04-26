import os
import models.fit_predict as fit_predict


def test_model_train(model_train, training_pipeline_params):
    assert (type(model_train).__name__ == training_pipeline_params.train_params.model_type)
    assert (vars(model_train).get('coef_', None) is not None)


def test_predict(predict_array, size_df):
    assert (len(predict_array.shape) == 1)
    assert (predict_array.shape[0] == size_df)


def test_metrics(predict_array, target):
    metrics = fit_predict.evaluate_model(predict_array, target)
    assert (isinstance(metrics, dict))


def test_cross_val_model(model_train, features, target):
    metrics = fit_predict.cross_val_model(model_train, features, target)
    assert (isinstance(metrics, dict))


def test_dump_model(model_train, tmpdir):
    path = tmpdir + 'tmp_model'
    fit_predict.dump_model(model_train, path)
    assert (os.path.exists(path))
