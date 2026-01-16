# AI in NBA Betting

This project predicts the total points scored in an NBA game using historical team performance data.

## Approach
- Linear Regression (baseline)
- Random Forest Regressor (final model)

## Evaluation
Models are evaluated using Mean Squared Error (MSE).

## How to Run
```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/model_training.py
python src/evaluate.py
