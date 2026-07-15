# 🏆 2026 FIFA World Cup Predictive Model

A lightweight machine learning pipeline that predicts World Cup knockout matches using a small set of high-signal features. Instead of relying on dozens of statistics, the model focuses on a parsimonious feature set to reduce overfitting and improve generalization.

## Results

- ✅ Correctly predicted **12 of 14** knockout matches
- ✅ Correctly predicted **7 of 8** quarterfinal matches
- ✅ Projected Spain's dominant semifinal performance over France with a **0.95 xG advantage**
- ✅ Generated pre-match projections for the remaining semifinal and final

## Methodology

### Features

Each feature is calculated as the difference between the two teams (Home − Away):

- Adjusted Market Value
- International Caps
- Adjusted Offensive Elo
- Adjusted Defensive Elo

All features are standardized using `StandardScaler` before training.

### Opponent Adjustment

Raw attacking and defensive statistics can be inflated by facing weaker opponents. To account for this, offensive and defensive ratings are weighted by opponent Elo, producing more representative measures of team strength.

### Model

- **Algorithm:** Ridge Regression (L2 Regularization)
- **Target:** Expected Goal Differential (Home xG − Away xG)

Ridge Regression was chosen because it performs well on smaller datasets while reducing the risk of overfitting through coefficient regularization.

## Repository Structure

```text
├── world_cup_prediction.ipynb   # Main notebook
├── README.md                    # Project documentation
└── data/                        # Match and team datasets
```

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Pipeline

1. Clean and merge tournament datasets.
2. Engineer opponent-adjusted features.
3. Create head-to-head feature differentials.
4. Scale features with `StandardScaler`.
5. Train a Ridge Regression model.
6. Evaluate feature importance.
7. Generate xG differential predictions for future matches.
