import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import pad_sequences
from tensorflow.python.keras.utils import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Model, Sequential, layers, regularizers, optimizers
from colorama import Fore, Style
import pickle
from typing import Tuple




# TODO: Do we really need classes???
#class Model:
#    def __init__(self,code):
#        self.code = code




def tokenize(X:np.ndarray):
    """
    Accepts raw source_code as input and tokenizes using TF-IDF
    :return: returns the preprocessed X and the vocab_size
    """
    # Initialize Tokenizer and fit to X
    tk = Tokenizer()
    tk.fit_on_texts(X)

    # Define vocab size
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')

    # Transform to seequences
    X_token = tk.texts_to_sequences(X)

    # Pad inputs
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', value=0)

    return X_pad, vocab_size



def initialize_model(X_pad: np.ndarray, y = np.ndarray, vocab_size = None) -> Model:
    """
    Initialize the CNN with random weights
    """

    print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

    if vocab_size is not None:
        input_dim = vocab_size
    else:
        print(f"\n❌ vocab size needed to define input dimension. Please insert the vocab size returned by the tokenize function.")
        return None

    input_length = X_pad.shape[1]

    print(f'input_dim = {input_dim}')
    print(f'input_length = {input_length}')

    model = Sequential([
        layers.Embedding(input_dim=input_dim, input_length=input_length, output_dim=256, mask_zero=True),
        layers.Conv1D(128, kernel_size=3),
        layers.MaxPool1D(pool_size = (4)),
        layers.Conv1D(128, kernel_size=5),
        layers.MaxPool1D(pool_size = (4)),
        layers.Conv1D(128, kernel_size=7),
        layers.MaxPool1D(pool_size = (4)),
        layers.Conv1D(128, kernel_size=9),
        layers.MaxPool1D(pool_size = (4)),
        layers.Flatten(),
        layers.Dense(len(np.unique(y)), activation="softmax"),  # check if we need to input the number of categories in softmax
        ])

    print("\n✅ model initialized: Summary:")
    print(model.summary())

    return model



def compile_model(model: Model, learning_rate: float) -> Model:
    """
    Compile the CNN
    """
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print("\n✅ model compiled")
    return model



def train_model(model: Model,
                X_pad: np.ndarray,
                y: np.ndarray,
                batch_size=64,
                epochs=200,
                patience=2,
                verbose=0,
                validation_split=0.2
                ) -> Tuple[Model, dict]:
    """
    Fit model and return a the tuple (fitted_model, history)
    """

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    # TODO Discuss: should we use monitor?
    es = EarlyStopping(patience=10,
                       restore_best_weights=True,
                       # monitor="val_loss"
                       )

    history = model.fit(X_pad,
                      y,
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=verbose,
                      validation_split=validation_split,
                      callbacks=[es])

    print(f"\n✅ model trained ({len(X_pad)} rows)")

    return model, history



def predict(code:np.ndarray)-> np.ndarray:
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """

    # Load model
    model = pickle.load(open("pipeline.pkl","rb"))

    # predict with model
    prediction = model.predict(code)

    # TODO inverse_transform the result
    return prediction
    # return prediction_inversed



def evaluate_model(model: Model,
                   X: np.ndarray,
                   y: np.ndarray,
                   batch_size=64) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset
    # TODO are the metrics rigt? Which one should we used?
    """

    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=1,
        # callbacks=None,
        return_dict=True)

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics
