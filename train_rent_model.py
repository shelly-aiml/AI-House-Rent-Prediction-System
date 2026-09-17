import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load student rent dataset
data = pd.read_csv("data/student_rent.csv")

# 2. Select input features
X = data[
    [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "distance_from_college_km",
        "furnished",
        "wifi",
        "food",
        "mess",
        "ac",
        "parking"
    ]
]

# 3. Select target
y = data["rent"]

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create model
model = LinearRegression()

# 6. Train model
model.fit(X_train, y_train)

# 7. Save trained model
joblib.dump(model, "models/student_rent_model.pkl")

print("Student rent model saved successfully!")

# 8. Make predictions
predictions = model.predict(X_test)

# 9. Evaluate model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===================================")
print("STUDENT RENT MODEL TRAINING")
print("===================================")
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 10. Predict rent for a new student room
new_room = pd.DataFrame({
    "area_sqft": [500],
    "bedrooms": [2],
    "bathrooms": [2],
    "distance_from_college_km": [2],
    "furnished": [1],
    "wifi": [1],
    "food": [1],
    "mess": [1],
    "ac": [1],
    "parking": [1]
})

predicted_rent = model.predict(new_room)

print("\n===================================")
print("NEW STUDENT RENT PREDICTION")
print("===================================")
print("Area:", 500, "sqft")
print("Bedrooms:", 2)
print("Bathrooms:", 2)
print("Distance:", 2, "km")
print("Predicted Monthly Rent: ₹", round(predicted_rent[0], 2))