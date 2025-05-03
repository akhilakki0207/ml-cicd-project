import joblib
import numpy as np

model = joblib.load("model/model.pkl")

def predict(input_data):
    data = np.array(input_data).reshape(1, -1)
    return model.predict(data)[0]