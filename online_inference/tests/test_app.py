import pytest
import sys, os
import json
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src')))
from app import app, load_model


@pytest.fixture(scope="session")
def correct_data_sample():
    return json.dumps([
        {
            "sex": 0,
            "cp": 0,
            "fbs": 0,
            "restecg": 1,
            "exang": 0,
            "slope": 2,
            "ca": 1,
            "thal": 3,
            "age": 40,
            "trestbps": 120,
            "chol": 240,
            "thalach": 150,
            "oldpeak": 0
        }, {
            "sex": 1,
            "cp": 1,
            "fbs": 0,
            "restecg": 1,
            "exang": 0,
            "slope": 2,
            "ca": 1,
            "thal": 2,
            "age": 78,
            "trestbps": 120,
            "chol": 240,
            "thalach": 150,
            "oldpeak": 0
        }
    ])


@pytest.fixture(scope="session")
def incorrect_data_sample():
    return json.dumps(
        [
            {
                "sex": 9,
                "cp": 0,
                "fbs": 0,
                "restecg": 1,
                "exang": 0,
                "slope": 2,
                "ca": 1,
                "thal": 3,
                "age": 40,
                "trestbps": 120,
                "chol": 240,
                "thalach": 150,
                "oldpeak": 0
            }
        ])


def test_api_prediction(correct_data_sample):
    with TestClient(app) as client:
        response = client.post('/predict/', data=correct_data_sample)
        assert response.status_code == 200

        result = [r['target'] for r in response.json()]
        assert result == [0, 1]


def test_incorrect_data_prediction(incorrect_data_sample):
    with TestClient(app) as client:
        response = client.post('/predict/', data=incorrect_data_sample)

        assert response.status_code == 400


def test_api_root():
    with TestClient(app) as client:
        response = client.get('/')

        assert response.status_code == 200


def test_api_touch():
    with TestClient(app) as client:
        response = client.get('/touch')

        assert response.status_code == 200
