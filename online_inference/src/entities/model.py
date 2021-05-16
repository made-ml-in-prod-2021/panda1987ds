'''Predict model'''
import pandas as pd
import pickle
from typing import List

from entities.response import Response
from entities.request_data_item import RequestDataItem


class Model:
    def __init__(self, input_path: str):
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
        self.data_pipeline = model['data_pipeline']
        self.classifier = model['classifier']
        self.ready_to_use = True

    def predict(self, request: List[RequestDataItem]) -> List[Response]:
        if self.ready_to_use:
            data_df = pd.DataFrame([item.__dict__ for item in request])
            data_df = pd.DataFrame(self.data_pipeline.transform(data_df))
            predictions = self.classifier.predict(data_df)
            return [Response(target=int(r)) for r in predictions]
        else:
            raise RuntimeError('The model is not loaded')
