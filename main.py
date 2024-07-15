import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn import set_config
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

from scripts.func import drop_missing, add_more_features, outlier_predictor
from scripts.pipe import RemoveCollinearColumns, CategoricalTransformer, TimeTransformer

train_df = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
train_df.head()

obsoleted_km_columns = [
    'metro_km_walk',
    'railroad_station_walk_km',
    'public_transport_station_km'
]
obsoleted_km_columns = np.concatenate((obsoleted_km_columns,
    train_df.loc[:, train_df.columns.str.contains('5000|3000|2000', case=False)].columns.to_numpy()))

ROUNDS = 450
params = {
	'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.06,
    'verbose': 1,
    'num_leaves': 2 ** 5,
    'bagging_fraction': 0.95,
    'bagging_freq': 1,
    'bagging_seed': 20231003,
    'feature_fraction': 0.7,
    'feature_fraction_seed': 20231003,
    'max_bin': 100,
    'max_depth': 8,
    'n_estimators': 150,
    'num_leaves': 35,
    'num_rounds': ROUNDS
}


set_config(transform_output='pandas')
pipe = Pipeline([
    ('drop_missing', FunctionTransformer(drop_missing(0.4))),
    ('add_more_features', FunctionTransformer(add_more_features)),
    ('drop_cols', make_column_transformer(
        ('drop', np.concatenate((obsoleted_km_columns, ['timestamp', 'sub_area']))),
        ('drop', make_column_selector(pattern=r'^ID_*')), remainder='passthrough', verbose_feature_names_out=False)),
    ('categorical_encoding', CategoricalTransformer()),
    ('simple_imputer', SimpleImputer(strategy='median')),
    ('remove_collinearity', VarianceThreshold(threshold=0.85)),
    ('scaler', StandardScaler()),
    ('light_gbm', LGBMRegressor(**params))
])

X_train = train_df.drop(columns=['price_doc'])
y_train = train_df['price_doc']
# pipe.fit(X_train, y_train)

param_grid = {
    'light_gbm__max_depth': [5, 6, 7],
    'light_gbm__learning_rate': [0.02, 0.03, 0.05],
    'light_gbm__n_estimators': [50, 100, 200],
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# # Fit to the data
grid_search.fit(X_train, y_train)

# # Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
predictions = best_model.predict(test_df)
print(predictions)