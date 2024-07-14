import pandas as pd

def drop_missing(threshold: float):
    def drop(df: pd.DataFrame):
        #Dropping columns with missing value rate higher than threshold
        df = df[df.columns[df.isnull().mean() < threshold]]

        #Dropping rows with missing value rate higher than threshold
        df = df.loc[df.isnull().mean(axis=1) < threshold]
        return df
    return drop

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

def outlier_predictor(model, X):
    return model.predict(X).reshape(-1, 1)