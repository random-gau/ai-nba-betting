import pandas as pd
import os

# Path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "nba_betting_data.csv")

# Load dataset
data = pd.read_csv(data_path)

# Create target variable: actual total points scored
data["actual_total_points"] = data["score_home"] + data["score_away"]

# Select sportsbook-based features
features = [
    "total",              # Over/Under line
    "spread",             # Point spread
    "moneyline_home",
    "moneyline_away"
]

X = data[features]
y = data["actual_total_points"]

# Drop rows with missing values
X = X.dropna()
y = y.loc[X.index]

print("Dataset loaded and preprocessed successfully!")
print("Number of samples:", len(X))
print(X.head())
