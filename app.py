from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models
house_model = joblib.load("models/house_price_model.pkl")
rent_model = joblib.load("models/student_rent_model.pkl")


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# House price prediction
@app.route("/house-price", methods=["GET", "POST"])
def house_price():

    prediction = None

    if request.method == "POST":

        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        floors = int(request.form["floors"])
        parking = int(request.form["parking"])

        input_data = pd.DataFrame({
            "area_sqft": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "floors": [floors],
            "parking": [parking]
        })

        prediction = house_model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template(
        "house_price.html",
        prediction=prediction
    )


# Student rent prediction
@app.route("/student-rent", methods=["GET", "POST"])
def student_rent():

    prediction = None

    if request.method == "POST":

        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        distance = float(request.form["distance"])
        furnished = int(request.form["furnished"])
        wifi = int(request.form["wifi"])
        food = int(request.form["food"])
        mess = int(request.form["mess"])
        ac = int(request.form["ac"])
        parking = int(request.form["parking"])

        input_data = pd.DataFrame({
            "area_sqft": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "distance_from_college_km": [distance],
            "furnished": [furnished],
            "wifi": [wifi],
            "food": [food],
            "mess": [mess],
            "ac": [ac],
            "parking": [parking]
        })

        prediction = rent_model.predict(input_data)[0]
        prediction = round(prediction, 2)

    return render_template(
        "student_rent.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)