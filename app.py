from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

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
    region = request.form['region']

    region_northwest = 1.0 if region == 'northwest' else 0.0
    region_southeast = 1.0 if region == 'southeast' else 0.0
    region_southwest = 1.0 if region == 'southwest' else 0.0

    features = pd.DataFrame(
        [[age, sex, bmi, children, smoker,
          region_northwest, region_southeast, region_southwest]],
        columns=['age', 'sex', 'bmi', 'children', 'smoker',
                 'region_northwest', 'region_southeast', 'region_southwest']
    )

    prediction = model.predict(features)
    prediction = np.expm1(prediction)

    return render_template(
        'index.html',
        prediction_text=f'Estimated Insurance Cost: ${prediction[0]:,.2f}',
        form_values=request.form
    )

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)