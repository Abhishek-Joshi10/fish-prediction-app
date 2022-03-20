
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import jsonify, render_template, url_for

app = Flask(__name__)
classifier = pickle.load(open('lab4.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=["POST"])
def predict():
    # print(classifier.predict([[250,11,25,13,2,4]]))
    values = [x for x in request.form.values()]
    print(values)
    array_value = [np.array(values)]
    prediction = classifier.predict(array_value)

    return render_template("home.html", prediction_text='Your fish type is {}'.format(str(prediction)))


if __name__ == '__main__':
    app.run()
