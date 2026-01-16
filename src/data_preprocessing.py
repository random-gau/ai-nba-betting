import pandas as pd
import os

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Build correct path to data file
data_path = os.path.join(BASE_DIR, "data", "nba_games.csv")

# Load dataset
data = pd.read_csv(data_path)

# Create target variable
data["total_points"] = data["home_points"] + data["away_points"]

print("Dataset loaded successfully!")
print(data.head())
