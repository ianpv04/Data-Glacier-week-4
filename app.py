import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        output = 'Diabetes predicted for given values!'
    else:
        output = 'No diabetes predicted for given values. :)'

    return render_template('index.html', prediction_text= output)

if __name__ == "__main__":
    app.run("0.0.0.0", port=5000)