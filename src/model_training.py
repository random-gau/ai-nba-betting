import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "nba_games.csv")

# Load data
data = pd.read_csv(data_path)

# Target variable
data["total_points"] = data["home_points"] + data["away_points"]

# Feature selection
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------
# Model 1: Linear Regression (Baseline)
# --------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)

# --------------------
# Model 2: Random Forest Regressor
# --------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Results
print("Model Performance (Mean Squared Error):")
print(f"Linear Regression MSE: {lr_mse:.2f}")
print(f"Random Forest MSE: {rf_mse:.2f}")
