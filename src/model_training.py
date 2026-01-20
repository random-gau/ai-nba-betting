import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "nba_betting_data.csv")

# Load dataset
data = pd.read_csv(data_path)

# Target variable
data["actual_total_points"] = data["score_home"] + data["score_away"]

# Sportsbook-based features
features = [
    "total",
    "spread",
    "moneyline_home",
    "moneyline_away"
]

X = data[features].dropna()
y = data.loc[X.index, "actual_total_points"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_preds)

# Random Forest (main model)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_preds)

print("Model Performance (Real Betting Dataset)")
print("----------------------------------------")
print(f"Linear Regression MSE: {lr_mse:.2f}")
print(f"Random Forest MSE: {rf_mse:.2f}")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
