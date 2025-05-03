from flask import Flask, request, jsonify
from src.predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    features = data['features']
    prediction = predict(features)
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)