from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, encoders, and scaler
model = pickle.load(open('heart_disease_model.sav', 'rb'))
encoders = pickle.load(open('heart_disease_encoders.sav', 'rb'))
scaler = pickle.load(open('heart_disease_scaler.sav', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = {}
    # Get form data
    data['age'] = float(request.form['age'])
    data['sex'] = request.form['sex'] # cat
    data['cp'] = request.form['cp'] #cat
    data['trestbps'] = float(request.form['trestbps'])
    data['chol'] = float(request.form['chol'])
    data['fbs'] = request.form['fbs']
    data['restecg'] = request.form['restecg'] #cat
    data['thalch'] = float(request.form['thalch'])
    data['exang'] = request.form['exang']
    data['oldpeak'] = float(request.form['oldpeak'])
    data['slope'] = request.form['slope'] #cat

    if data['fbs'] =='Yes':
        data['fbs']=1.0
    else:
        data['fbs']=0.0

    if data['exang'] == 'Yes':
        data['exang'] = 1.0
    else:
        data['exang'] = 0.0

    df = pd.DataFrame([data])

    for i in encoders['sex'].categories_[0]:
        df['sex' + '_' + i] = 0.0
    df['sex' + '_' + df['sex']] = 1.0
    df.drop(columns='sex', inplace=True)

    for i in encoders['cp'].categories_[0]:
        df['cp' + '_' + i] = 0.0
    df['cp' + '_' + df['cp']] = 1.0
    df.drop(columns='cp', inplace=True)


    for i in encoders['restecg'].categories_[0]:
        df['restecg' + '_' + i] = 0.0
    df['restecg' + '_' + df['restecg']] = 1.0
    df.drop(columns='restecg', inplace=True)

    for i in encoders['slope'].categories_[0]:
        df['slope' + '_' + i] = 0.0
    df['slope' + '_' + df['slope']] = 1.0
    df.drop(columns='slope', inplace=True)

    df =pd.DataFrame(scaler.transform(df),columns=df.columns)


    # Make prediction
    prediction = model.predict(df)

    return f'Predicted result: {prediction}'


if __name__ == '__main__':
    app.run(debug=True)



