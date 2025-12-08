from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    score = float(request.form["score"])

    X = np.array([[hours, score]])
    X = scaler.transform(X)

    prediction = model.predict(X)[0]
    result = "PASS" if prediction == 1 else "FAIL"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)