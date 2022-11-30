import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import xgboost as xgb
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import pickle
from colorama import Fore, Style
import os




def preprocess_data(data:pd.DataFrame)->pd.DataFrame:
    # Remove NAs
    data = data.dropna()
    # Remove code with less than x characters
    data = data.loc[data['code_source'].str.len() > 5]
    # Remove users with entries < 25
    data["username"].value_counts()
    #data = data[data['username'].map(data['username'].value_counts()) > 25].reset_index(drop = True)
    # when there are more than 1 submissions, keep only the last one
    data = data.drop_duplicates(subset=['year', 'round', 'username', 'task'], keep='first')
    return data


def encode_y(data:pd.DataFrame)->pd.DataFrame:
    target_encoder = LabelEncoder().fit(data['username'])
    y = target_encoder.transform(data['username'])
    return y, target_encoder

def create_model():
    pipeline_svc = make_pipeline(
    TfidfVectorizer(),
    LinearSVC()
    )
    return pipeline_svc

def tune_model(model = None, X_train = None, X_test = None, y_train = None, y_test = None):
    bs_opt = BayesSearchCV(
    model,
     {
         'linearsvc__C': Real(low=0.001, high=10, prior='log-uniform', transform='identity'),
         'tfidfvectorizer__min_df': Integer(low=0, high=150, prior='uniform'),
         'tfidfvectorizer__max_df': Real(low=0.2, high=0.35, prior='uniform'),
         #'tfidfvectorizer__ngram_range':  Categorical([(1,1), (1,2)])
         #'tfidfvectorizer__ngram_range': Categorical([(1,1), (1,2), (1,3), (1,4), (1,5),(2, 2), (3,3), (4,4), (5,5)])
     },
     n_iter=32,
     random_state=0
    )

    # Execute Bayesian OPtimization
    res = bs_opt.fit(X_train, y_train)

    # Perform Grid Search
    grid_search = GridSearchCV(
        model,
        {
        'tfidfvectorizer__ngram_range': [(2,2), (3, 3), (4, 4), (5, 5), (1, 2), (1, 3), (1, 4), (1, 5)],
        'linearsvc__C': [bs_opt.best_params_["linearsvc__C"]],
        'tfidfvectorizer__min_df': [bs_opt.best_params_["tfidfvectorizer__min_df"]],
        'tfidfvectorizer__max_df': [bs_opt.best_params_["tfidfvectorizer__max_df"]],
        },
        cv = 5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best params
    print(f"Best Score = {grid_search.best_score_}")
    print(f"Best params = {grid_search.best_params_}")

    # Print final score
    print(grid_search.score(X_test, y_test))

    bs_opt_tuned = grid_search.best_estimator_

    return bs_opt_tuned


def save_model(model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join('params.pkl', "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join('metrics.pkl', "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join('model.pkl', "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ Data saved locally")

    return None



def predict(code:str)-> np.ndarray:
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    print(Fore.BLUE + "\nPredict author..." + Style.RESET_ALL)

    # Load model
    model = pickle.load(open("models/linearsvc.pkl","rb"))

    # predict with model
    prediction = model.predict(np.array([code], dtype=object))
    print(prediction)
    # TODO inverse_transform the result
    print(f"\n✅ Prediction done!")
    return prediction


#data = pd.read_csv('raw_data/preprocessed_dataset.csv')[:1000]

#data = preprocess_data(data = data)

#y, target_encoder = encode_y(data = data)

#X = data["code_source"]

predict("Lass uns diesen Code mal testen")
