from fastapi import FastAPI

app = FastAPI()

# Root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

# Predict route
@app.get('/predict_author')
def predict():
    return {'author': 'YOU'}
