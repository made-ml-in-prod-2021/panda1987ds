'''Application entry point'''
import os.path

from typing import List
import logging.config
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import time

from entities.model import Model
from entities.request_data_item import RequestDataItem
from entities.response import Response

os.chdir(os.path.abspath(os.path.dirname(__file__)))
app = FastAPI()
logging.config.fileConfig('../configs/logging.ini', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
model = None
MODEL_PATH = "../models/model.pkl"
PORT = 8080
BREAK_TIME = 120
SLEEP_TIME = 30
IP = '0.0.0.0'


@app.on_event("startup")
def load_model():
    global model
    logger.info('Start sleep')
    time.sleep(SLEEP_TIME)
    global start_time
    start_time = time.time()
    logger.info('Start load model')
    try:
        model = Model(MODEL_PATH)
        logger.info('Model loaded successfully')
    except FileNotFoundError():
        logger.error("Critical error. Can't load model")
        raise HTTPException(status_code=500, detail="Model file not found")


@app.get('/touch')
def touch() -> bool:
    global model
    if time.time() - start_time > BREAK_TIME:
        logger.error(f'Time is over')
        model = None
    return model is not None and model.ready_to_use


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.get('/')
def main():
    if touch():
        return 'Please use the command "predict"'
    else:
        raise HTTPException(status_code=500, detail="Not ready to use")


@app.post('/predict/', response_model=List[Response])
def predict(request: List[RequestDataItem]) -> List[Response]:
    if touch():
        try:
            logger.info('Start request...')
            predicts = model.predict(request)
            logger.info('Request completed successfully')

            return predicts

        except Exception as ex:
            logger.error(f'Critical model error: {str(ex)}')
            raise HTTPException(status_code=500,
                                detail="Critical error. Can't make a prediction")
    else:
        logger.error(f'Not ready to use')
        raise HTTPException(status_code=500,
                                detail="Critical error. Can't make a prediction. Not ready to use")


def main():
    load_model()
    uvicorn.run(app, host=IP, port=PORT)


if __name__ == "__main__":
    main()
