import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer, MinMaxScaler, RobustScaler

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