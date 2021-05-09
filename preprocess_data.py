import pandas as pd
import numpy as np
import ast
from constants import *


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


def prepare_df_for_baseline(df: pd.DataFrame):
    columns_not_included = ['backdrop_path', 'homepage', 'id', 'imdb_id', 'poster_path', 'production_countries',
                            'spoken_languages', 'status', 'video']
    textual_columns = ['original_title', 'overview', 'tagline', 'title', 'Keywords']
    target_column = 'revenue'

    y = np.array(df[target_column])
    X = df.drop(columns_not_included + textual_columns + [target_column], axis=1)

    columns_to_dict = ['belongs_to_collection', 'genres', 'production_companies', 'cast', 'crew']
    for col in columns_to_dict:
        X[col] = X[col].apply(lambda x: ast.literal_eval(str(x)) if type(x) == str else x)

    X['budget'].replace({0: np.nan}, inplace=True)
    X["release_date"] = X["release_date"].astype("datetime64")
    X['runtime'].replace({0: np.nan}, inplace=True)

    X['belongs_to_collection'] = X['belongs_to_collection'].apply(
        lambda x: x.get('name', '<no collection>') if type(x) == dict else '<no collection>')

    # only first genre
    X['genres'] = X['genres'].apply(
        lambda x: x[0].get('name', '<no genre>') if type(x) == list and len(x) > 0 and type(x[0]) == dict
        else '<no genre>')

    # only first production company
    X['production_companies'] = X['production_companies'].apply(
        lambda x: x[0].get('name', '<no company>') if type(x) == list and len(x) > 0 and type(x[0]) == dict
        else '<no company>')

    # take only year as int
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

    X = X.drop(['cast', 'crew'], axis=1)

    return X, y


def load_data_for_baseline_ml():
    """
    loads the train and test data ready for the baseline settings (features)
    :return: (X_train, y_train), (X_test, y_test). X is pd.DataFrame, y is pd.Series
    """
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')
    return prepare_df_for_baseline(train_df), prepare_df_for_baseline(test_df)


if __name__ == '__main__':
    train = pd.read_csv(TRAIN_PATH, sep='\t')
    test = pd.read_csv(TEST_PATH, sep='\t')
