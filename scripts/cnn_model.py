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
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras import Model, Sequential, layers, regularizers, optimizers
from tensorflow.keras.utils import to_categorical
from colorama import Fore, Style
import pickle
from typing import Tuple
import os
import time
import glob
import models



def tokenize(X:np.ndarray):
    """
    Accepts raw source_code as input and tokenizes using TF-IDF
    :return: returns the preprocessed X and the vocab_size
    """
    print(Fore.BLUE + "\nTokenizes source code..." + Style.RESET_ALL)
    # Initialize Tokenizer and fit to X
    tk = Tokenizer()
    tk.fit_on_texts(X)

    # Define vocab size
    vocab_size = len(tk.word_index)
    print(f'There are {vocab_size} different words in your corpus')

    # Transform to sequences
    X_token = tk.texts_to_sequences(X)

    # Pad inputs
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', value=0)
    print("\n✅ Source code tokenized")
    return X_pad, vocab_size



def label_encode(y:np.ndarray)->np.ndarray:
    target_encoder = LabelEncoder().fit(y)
    target_encoded = target_encoder.transform(y)

    target_cat = to_categorical(target_encoded,
                                num_classes = len(np.unique(target_encoded)))

    return target_cat


def initialize_model(X_pad: np.ndarray,
                     y = np.ndarray,
                     vocab_size = None) -> Model:
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
        layers.Dense(y.shape[1], activation="softmax"),  # check if we need to input the number of categories in softmax
        ])

    print("\n✅ Model initialized. Summary:")
    print(model.summary())

    return model



def compile_model(model: Model) -> Model:
    """
    Compile the CNN
    """
    print(Fore.BLUE + "\nCompile model..." + Style.RESET_ALL)
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    print("\n✅ Model compiled")
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
    Fit model and return a tuple (fitted_model, history)
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

    print(f"\n✅ Model trained ({len(X_pad)} rows)")

    return model, history


def predict_cnn(code:str):
    """
    Accepts a piece of code as an input, to predict its author as a return.
    :param code: a given peace of code.
    :return: returns an array containing one or more predictions of authors for the given peaces of code
    """
    print(Fore.BLUE + "\nPredict author..." + Style.RESET_ALL)

    # Load model
    model = pickle.load(open("models/tfidf_nn.pkl","rb"))
    # load tfdidf vectorizer
    tfidf_vectorizer = pickle.load(open("models/tfidf_vec.pkl","rb"))

    # load label_encoder
    target_encoder = pickle.load(open("models/nn_target_encoder.pkl","rb"))

    # tfidf-transform code
    code_tfidf = tfidf_vectorizer.transform([code])
    code_tfidf = code_tfidf.astype("float32")

    # predict with model
    prediction_proba=model.predict(code_tfidf)
    prediction = target_encoder.inverse_transform([np.argmax(pred) for pred in prediction_proba])

    prediction_proba_list = dict(zip(target_encoder.classes_, prediction_proba[0].tolist()))

    # get 5 highest probabilities
    x=list(prediction_proba_list.values())
    top5=dict()
    x.sort(reverse=True)
    x=x[:5]
    for i in x:
        for j in prediction_proba_list.keys():
            if(prediction_proba_list[j]==i):
                top5[j] = prediction_proba_list[j]

    print(f"\n✅ Prediction done!")
    print(prediction)
    print(top5)

    return str(prediction[0]) , top5
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

    print(f"\n✅ Model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

    return metrics


def save_model(model: Model = None,
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


# TODO not working yet - check paths?
def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join("models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model



# read data
#data= pd.read_csv('raw_data/preprocessed_dataset.csv')[:1000]
#X = data["code_source"]
#y = label_encode(y = data["username"])
#data_tokenized, vocab_size = tokenize(X = X)
#print(data_tokenized)
#print(vocab_size)##

#model = initialize_model(X_pad=data_tokenized,
#                         y = y,
#                         vocab_size = vocab_size)


#compile_model(model = model)


#model, history = train_model(model = model,
#                X_pad = data_tokenized,
#                y = y)

#save_model(model = model)

#metrics = evaluate_model(model= model,
#                   X= X_test,
#                   y= y_test,
#                   batch_size=64)

#sourcecode = """
#
#    }
#}
#
#int main()
#{
#    int T;
#
#    cin >> T;
#    for (int ct = 0; ct < T; ++ct)
#    {
#        int r, c, n, d;
#        cin >> r >> c >> n >> d;
#        vector<vector<long long>> v(r, vector<long long>(c, 1000000000000000000ll));
#        vector<vector<bool>> fixed(r, vector<bool>(c, false));
#
#        for (int i = 0; i < n; ++i)
#        {
#            int x, y;
#            long long z;
#            cin >> x >> y >> z;
#            x--;
#            y--;
#            v[x][y] = z;
#            fixed[x][y] = true;
#        }
#
#       """
#
#a, b = predict_cnn(sourcecode)
#
#print(type(a))
#print(type(b))


#data = pd.read_csv('raw_data/preprocessed_dataset.csv')
#target_encoder = LabelEncoder().fit(data['username'])
#y = target_encoder.transform(data['username'])
#pickle.dump(target_encoder, open(os.path.join(os.getcwd(),'models/nn_target_encoder.pkl'), 'wb'))
