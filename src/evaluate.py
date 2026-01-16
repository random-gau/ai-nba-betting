import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "nba_games.csv")

# Load data
data = pd.read_csv(data_path)
data["total_points"] = data["home_points"] + data["away_points"]

features = [
    "home_rebounds",
    "away_rebounds",
    "home_assists",
    "away_assists",
    "home_turnovers",
    "away_turnovers"
]

X = data[features]
y = data["total_points"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train final model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Final Model Evaluation")
print("----------------------")
print(f"Mean Squared Error: {mse:.2f}")

print("\nPredictions vs Actual:")
for pred, actual in zip(predictions, y_test):
    print(f"Predicted: {pred:.1f}, Actual: {actual}")
