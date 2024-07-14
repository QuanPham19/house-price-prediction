import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import set_config
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train_df = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
train_df.head()

def drop_missing(threshold: float):
    def drop(df: pd.DataFrame):
        #Dropping columns with missing value rate higher than threshold
        df = df[df.columns[df.isnull().mean() < threshold]]

        #Dropping rows with missing value rate higher than threshold
        df = df.loc[df.isnull().mean(axis=1) < threshold]
        return df
    return drop

# drop these as the versions of "Time to" columns is strongly correlated with these
obsoleted_km_columns = [
    'metro_km_walk',
    'railroad_station_walk_km',
    'public_transport_station_km'
]
obsoleted_km_columns = np.concatenate((obsoleted_km_columns,
    train_df.loc[:, train_df.columns.str.contains('5000|3000|2000', case=False)].columns.to_numpy()))

def add_more_features(df: pd.DataFrame):
    df['population_per_area'] = df['raion_popul'] / df['area_m']

    df = df.drop(columns=['raion_build_count_with_material_info', 'build_count_block',
                            'build_count_wood', 'build_count_frame',
                            'build_count_brick', 'build_count_monolith',
                            'build_count_panel', 'build_count_foam',
                            'build_count_slag', 'build_count_mix',
                            'ID_railroad_station_walk', 'ID_railroad_station_avto',
                            'ID_big_road1', 'ID_big_road2',
                            'hospital_beds_raion'], axis=1, errors='ignore')

    df['room_per_sq'] = df['life_sq'] / (df['num_room'] + 1)
    df['floor_per_max'] = df['floor'] / (df['max_floor'] + 1)

    df['pop_per_mall'] = df['shopping_centers_raion'] / df['raion_popul']
    df['pop_per_office'] = df['office_raion'] / df['raion_popul']

    df['preschool_fill'] = df['preschool_quota'] / df['children_preschool']
    df['preschool_capacity'] = df['preschool_education_centers_raion'] / df['children_preschool']
    df['school_fill'] = df['school_quota'] / df['children_school']
    df['school_capacity'] = df['school_education_centers_raion'] / df['children_school']

    df['percent_working'] = df['work_all'] / df['full_all']
    df['percent_old'] = df['ekder_all'] / df['full_all']

    return df

class RemoveCollinearColumns:
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.high_corr_cols = set()

    def fit(self, X, y=None):
        # Convert the NumPy array to a DataFrame
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= self.threshold:
                    colname = corr_matrix.columns[i]
                    self.high_corr_cols.add(colname)

        return self

    def transform(self, X):
        # Convert the NumPy array to a DataFrame
        X_df = pd.DataFrame(X)
        X_copy = X_df.copy()

        # Drop columns identified during fitting
        X_copy.drop(columns=self.high_corr_cols, axis=1, inplace=True)

        return X_copy.to_numpy()


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

class CategoricalTransformer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X)
        cat_columns = X.select_dtypes(exclude=["number","bool_"]).columns
        cat_columns = cat_columns.drop(['timestamp', 'sub_area', 'ecology'], errors='ignore')
        X[cat_columns] = X[cat_columns].apply(lambda x: x.str.replace('yes', '1'))
        X[cat_columns] = X[cat_columns].apply(lambda x: x.str.replace('no', '0'))

        X['product_type'] = X['product_type'].apply(lambda x: str(x).replace('Investment', '1'))
        X['product_type'] = X['product_type'].apply(lambda x: str(x).replace('OwnerOccupier', '0'))
        encoder = OrdinalEncoder()
        X['ecology'] = encoder.fit_transform(X[['ecology']])
        X[cat_columns] = X[cat_columns].astype('Float64')
        return X

class TimeTransformer:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = pd.DataFrame(X)
        X['timestamp'] = pd.to_datetime(X['timestamp'])
        X['timestamp'] = X['timestamp'].dt.to_period('M')
        no_period = X.drop(['timestamp'], axis=1).columns
        X[no_period] = X[no_period].fillna(X[no_period].median())
        return X

from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import set_config
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

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

def outlier_predictor(model, X):
    return model.predict(X).reshape(-1, 1)

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
pipe.fit(X_train, y_train)

test_df = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
predictions = pipe.predict(test_df)
print(predictions)