# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('boston_housing_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    data = request.get_json(force=True)
    prediction = model.predict(np.array(data['features']).reshape(1, -1))
    return jsonify(prediction=prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
