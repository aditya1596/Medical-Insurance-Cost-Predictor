from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = float(request.form['smoker'])
    region = float(request.form['region'])

    # Input array
    features = np.array([[age, sex, bmi, children, smoker, region]])

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)
    prediction = np.expm1(prediction)

    return render_template(
        'index.html',
        prediction_text=f'Estimated Insurance Cost: ₹{prediction[0]:,.2f}'
    )

# Run app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)