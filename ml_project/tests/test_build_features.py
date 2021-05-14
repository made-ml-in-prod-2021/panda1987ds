import pandas as pd


def test_build_features(features, size_df,numeric_features, sq_features):
    assert isinstance(features, pd.DataFrame)
    assert features.shape == (size_df, 29)
    assert features.isnull().sum().all() == 0
    assert round(features.iloc[:, -(len(numeric_features)+len(sq_features))].sum()) == 0
    assert round(features.iloc[:, :-(len(numeric_features)+len(sq_features))].min()).all() == 0
    assert round(features.iloc[:, :-(len(numeric_features)+len(sq_features))].max()).all() == 1


def test_target(target, size_df):
    assert len(target.shape) == 1
    assert target.shape[0] == size_df
