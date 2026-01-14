import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("prediction.csv")

# Features & target
X = data[['Height', 'Weight']]
y = data['Age']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train ML model
model = LinearRegression()
model.fit(X_train, y_train)

# -------- USER INPUT --------
height = float(input("Enter Height (cm): "))
weight = float(input("Enter Weight (kg): "))

# Predict age
predicted_age = model.predict([[height, weight]])

print(f"\nPredicted Age: {int(predicted_age[0])} years")
