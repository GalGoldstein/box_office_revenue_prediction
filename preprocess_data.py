"""
This script contains function to pre-process the data for machine learning
"""

import pandas as pd
import numpy as np
import ast
from constants import *
from df2numpy import TransformDF2Numpy  # https://github.com/yotammarton/TransformDF2Numpy
from pickle import dump


def _get_first_crew_member_per_job(crew_list: list, jobs: list):
    """
    return a dict with crew members names (that appear first) for every job in `jobs`
    :param crew_list: (list) a list of crew members data
    :param jobs: (list) a list of wanted jobs
    :return: dict with crew members names for every job in `jobs`
    """
    result = dict.fromkeys(jobs, '<no crew>')
    for job in jobs:
        for member_dict in crew_list:
            if member_dict.get('job') == job:
                result[job] = member_dict.get('name', '<no crew>')
                break  # to get the first only
    return result


def prepare_df_for_baseline(df: pd.DataFrame, zerotonan: bool = True):
    columns_not_included = ['backdrop_path', 'homepage', 'id', 'imdb_id', 'poster_path', 'production_countries',
                            'spoken_languages', 'status', 'video']
    textual_columns = ['original_title', 'overview', 'tagline', 'title', 'Keywords']
    target_column = 'revenue'

    y = np.array(df[target_column])
    X = df.drop(columns_not_included + textual_columns + [target_column], axis=1)

    columns_to_dict = ['belongs_to_collection', 'genres', 'production_companies', 'cast', 'crew']
    for col in columns_to_dict:
        X[col] = X[col].apply(lambda x: ast.literal_eval(str(x)) if type(x) == str else x)

    if zerotonan:  # uncertain data to nan
        X['budget'].replace({0: np.nan}, inplace=True)
        X['runtime'].replace({0: np.nan}, inplace=True)

    X["release_date"] = X["release_date"].astype("datetime64")

    X['belongs_to_collection'] = X['belongs_to_collection'].apply(
        lambda x: x.get('name', '<no collection>') if type(x) == dict else '<no collection>')

    # genres (first 3 if available)
    X['genre_1'] = X['genres'].apply(
        lambda x: x[0].get('name', '<no genre>') if type(x) == list and len(x) > 0 and type(x[0]) == dict
        else '<no genre>')
    X['genre_2'] = X['genres'].apply(
        lambda x: x[1].get('name', '<no genre>') if type(x) == list and len(x) > 1 and type(x[1]) == dict
        else '<no genre>')
    X['genre_3'] = X['genres'].apply(
        lambda x: x[2].get('name', '<no genre>') if type(x) == list and len(x) > 2 and type(x[2]) == dict
        else '<no genre>')

    # production companies (first 3 if available)
    X['production_company_1'] = X['production_companies'].apply(
        lambda x: x[0].get('name', '<no company>') if type(x) == list and len(x) > 0 and type(x[0]) == dict
        else '<no company>')
    X['production_company_2'] = X['production_companies'].apply(
        lambda x: x[1].get('name', '<no company>') if type(x) == list and len(x) > 1 and type(x[1]) == dict
        else '<no company>')
    X['production_company_3'] = X['production_companies'].apply(
        lambda x: x[2].get('name', '<no company>') if type(x) == list and len(x) > 2 and type(x[2]) == dict
        else '<no company>')

    # take  year and month as int
    X['release_month'] = X['release_date'].apply(lambda x: x.date().month)
    X['release_date'] = X['release_date'].apply(lambda x: x.date().year)

    # get 5 first cast members
    for i in range(5):
        X[f'cast_{i + 1}'] = X['cast'].apply(
            lambda x: x[i].get('name', '<no cast>')
            if type(x) == list and len(x) > i and type(x[i]) == dict else '<no cast>')

    # get first crew member in each job for every job in `jobs`
    jobs = ['Executive Producer', 'Producer', 'Director', 'Screenplay', 'Author']
    X['crew'] = X['crew'].apply(lambda x: _get_first_crew_member_per_job(x, jobs))

    for job in jobs:
        X[job] = X['crew'].apply(lambda x: x.get(job, '<no crew>'))

    X = X.drop(['cast', 'crew', 'genres', 'production_companies'], axis=1)

    return X, y


def load_data_for_baseline_ml():
    """
    loads the train and test data ready for the baseline settings (features)
    :return: (X_train, y_train), (X_test, y_test). X is pd.DataFrame, y is pd.Series
    """
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')
    return prepare_df_for_baseline(train_df), prepare_df_for_baseline(test_df)


def load_data_transform2DFNumpy(scale_popularity: bool = False, return_trans: bool = False):
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')

    X_train, y_train = prepare_df_for_baseline(df=train_df, zerotonan=True)
    X_test, y_test = prepare_df_for_baseline(df=test_df, zerotonan=True)

    X_train['revenue'] = y_train
    X_test['revenue'] = y_test

    if scale_popularity:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(1, 70))
        scaler.fit(np.array(X_train[(X_train['release_date'] == 2019)]['popularity']).reshape(-1, 1))
        # dump(scaler, open('scaler.pkl', 'wb'))  # TODO

        for df in [X_train, X_test]:
            for i, row in df.iterrows():
                if row['release_date'] == 2019:
                    new_popularity = np.float64(scaler.transform(np.array([df.at[i, 'popularity']]).reshape(-1, 1)))
                    df.at[i, 'popularity'] = new_popularity

    min_category_count_dict = dict.fromkeys(X_train.columns, 4)
    # min_category_count_dict['belongs_to_collection'] = 0

    trans = TransformDF2Numpy(objective_col='revenue',
                              fillnan=False,
                              numerical_scaling=True,
                              copy=True,
                              min_category_count=min_category_count_dict)
    X_train, y_train = trans.fit_transform(X_train)
    # dump(trans, open('trans.pkl', 'wb'))  # TODO
    X_test, y_test = trans.transform(X_test)

    if return_trans:
        return (X_train, y_train), (X_test, y_test), trans.num_categoricals, trans
    else:
        return (X_train, y_train), (X_test, y_test), trans.num_categoricals


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_PATH, sep='\t')
    test = pd.read_csv(TEST_PATH, sep='\t')
    # load_data_transform2DFNumpy(scale_popularity=True)
