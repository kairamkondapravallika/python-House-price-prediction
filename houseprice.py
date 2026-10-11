
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("house_prices.csv")

print("First five records:")
print(data.head())

# 2. Separate input features and target
X = data[
    ["area_sqft", "bedrooms", "bathrooms", "location_score"]
]
y = data["price_lakhs"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Train the Machine Learning model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict prices for test data
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mae, 2), "lakhs")
print("R2 Score:", round(r2, 3))

# 7. Get details of a new house
area = float(input("\nEnter house area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
location_score = float(input("Enter location score (1-10): "))

# 8. Predict the new house price
new_house = pd.DataFrame([{
    "area_sqft": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "location_score": location_score
}])

predicted_price = model.predict(new_house)[0]

print("\nEstimated House Price:")
print(f"Rs. {predicted_price:.2f} lakhs")

# 9. Visualize actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("House Price Prediction")
plt.tight_layout()
plt.show()
