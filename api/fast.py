from fastapi import FastAPI
from scripts.svc_model import predict_svc
from scripts.cnn_model import predict_cnn
from scripts.nn_model import predict_nn

app = FastAPI()

# Root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

# Predict route
@app.get('/predict_with_svc')
def predict_svc_api(code:str):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    author, prob = predict_svc(code)
    return {'author': author, 'probabilities': prob}

@app.get('/predict_with_cnn')
def predict_cnn_api(code:str):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    author, prob, tfidf_df = predict_cnn(code)
    return {'author': author, 'probabilities': prob, 'top_terms': tfidf_df}

@app.get('/predict_with_nn')
def predict_nn_api(code:dict):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    author, prob, tfidf_df = predict_nn(code["code"])
    return {'author': author, 'probabilities': prob, 'top_terms': tfidf_df}
