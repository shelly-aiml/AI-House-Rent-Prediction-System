
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
data = pd.read_csv("data/house_prices.csv")

# 2. Select input features
X = data[["area_sqft", "bedrooms", "bathrooms", "floors", "parking"]]

# 3. Select target
y = data["price"]

# 4. Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create the machine learning model
model = LinearRegression()

# 6. Train the model
model.fit(X_train, y_train)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "models/house_price_model.pkl")

print("Model saved successfully!")


# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("===================================")
print("MODEL TRAINING COMPLETED!")
print("===================================")

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# 9. Predict price for a new house

new_house = pd.DataFrame({
    "area_sqft": [1800],
    "bedrooms": [3],
    "bathrooms": [3],
    "floors": [2],
    "parking": [2]
})

predicted_price = model.predict(new_house)

print("\n===================================")
print("NEW HOUSE PREDICTION")
print("===================================")
print("Area:", 1800, "sqft")
print("Bedrooms:", 3)
print("Bathrooms:", 3)
print("Floors:", 2)
print("Parking:", 2)
print("Predicted Price: ₹", round(predicted_price[0], 2))