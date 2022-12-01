from fastapi import FastAPI
import os
import pickle
from colorama import Fore, Style
import numpy as np
from scripts.svc_model import predict_svc
from scripts.cnn_model import predict_cnn

app = FastAPI()

# Root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

# Predict route
@app.get('/Predict_author_with_SVC ')
def predict_svc_api(code:str):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    author, prob = predict_svc(code)
    import json

    # parse x:
    #y = json.dumps({'author': author})

    return {'author': author, 'probabilities': prob}
    #return {'author': author, 'probabilities': prob.tolist()}

@app.get('/predict_author_with_NN')
def predict_cnn_api(code:str):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    import json

    author, prob = predict_cnn(code)
    return {'author': author, 'probabilities': prob}
