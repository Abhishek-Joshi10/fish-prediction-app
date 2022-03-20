
import numpy as np
import pickle
import pandas as pd
from flask import Flask, request

app = Flask(__name__)
pickle_in = open("lab4.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Testing the model in local"


@app.route('/predict')
def predict_class():
    Weight = request.args.get('Weight')
    Length1 = request.args.get('Length1')
    Length2 = request.args.get('Length2')
    Length3 = request.args.get('Length3')
    Height = request.args.get('Height')
    Width = request.args.get('Width')
    prediction = classifier.predict([[Weight, Length1, Length2, Length3, Height, Width]])
    return " The Predicated Class is" + str(prediction)

@app.route('/prediction_model',methods=["Post"])
def prediction_model():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return " The predicted type of fish is " + str(list(prediction))


if __name__=='__main__':
    app.run()