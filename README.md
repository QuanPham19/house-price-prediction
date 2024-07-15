# house-price-prediction

## Overview
This is a forecasting project of house price leveraging Machine Learning techniques. We achieved top 10% Kaggle leaderboard after 3 months researching and working in a team of 4. Some highlights are indicated as below. Futher details are included in the PDF file.

The Russian housing market is one of the most dynamic and volatile in the world, influenced by various economic, political, and social factors. Accurately predicting the realty prices in Russia can help investors, home buyers, and policymakers make informed decisions and optimize their outcomes. 

### 1. Exploratory data analysis:
- Missing values handling
- Correlation exploration and heatmap visualization
- Distribution of column data

### 2. Feature engineering:
- Drop less meaningful and collinearity columns
- Generate crossed variables based on domain knowledge
- Remove outliers and low variance data

### 3. Model building:
- Leverage LightGBM, XGBoost and Catboost  
- Apply GridSearch and cross-validation
- Ensemble model by weighted average

### 4. Run prediction:
```
-- create virtual environment
python -m venv venv

-- activate virtual environment
source venv/bin/activate

-- install dependencies
pip install -r requirements.txt

-- process main file
python main.py
```