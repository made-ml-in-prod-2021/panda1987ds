
import configs.load_params as load_params


def test_params(training_pipeline_params):
    assert isinstance(training_pipeline_params, load_params.TrainingPipelineParams)
    assert isinstance(training_pipeline_params.feature_params, load_params.FeatureParams)
    assert isinstance(training_pipeline_params.splitting_params, load_params.SplittingParams)

