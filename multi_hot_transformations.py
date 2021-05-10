import preprocess_data
import numpy as np
import pandas as pd
from constants import *
import ast


def prepare_test(df: pd.DataFrame, dicts):
    columns_not_included = ['backdrop_path', 'homepage', 'id', 'imdb_id', 'poster_path', 'production_countries',
                            'spoken_languages', 'status', 'video']
    textual_columns = ['original_title', 'overview', 'tagline', 'title', 'Keywords']

    X = df.drop(columns_not_included + textual_columns, axis=1)

    columns_to_dict = ['genres', 'belongs_to_collection', 'cast',  'production_companies', 'crew']
    for col in columns_to_dict:
        X[col] = X[col].apply(lambda x: ast.literal_eval(str(x)) if type(x) == str else x)

    X['budget'] = X['budget'].apply(lambda x: x if x >= 2500 else np.nan)

    X["release_date"] = X["release_date"].astype("datetime64")
    X['release_date'] = X['release_date'].apply(lambda x: x.date().year)  # take only year as int

    X['runtime'].replace({0: np.nan}, inplace=True)
    X['belongs_to_collection'] = X['belongs_to_collection'].apply(lambda x: [x] if type(x) in [dict] else x)

    for col, dictionary in zip(columns_to_dict[:-1], dicts[:-1]):
        X[col] = X[col].apply(lambda x: extract_all_names(x) if type(x) in [dict, list] else [])
        X[col] = X[col].apply(lambda row: create_multi_hot(row, dictionary))

    X['crew'] = X['crew'].apply(lambda x: extract_all_names_for_crew(x) if type(x) in [dict, list] else [])
    X['crew'] = X['crew'].apply(lambda row: create_multi_hot(row, dicts[-1]))

    prefixes_names = ['genre', 'collection', 'actor',  'company', 'crew']
    X = X.reset_index(drop=True)
    for col, dictionary, prefix in zip(columns_to_dict, dicts, prefixes_names):
        mat = np.vstack(list(X[col].values))
        columns_names = [prefix + '_' + i for i in dictionary.keys()]
        df2 = pd.DataFrame(mat, columns=columns_names)
        X = pd.concat([X, df2], axis=1)

    X = X.drop(columns_to_dict, axis=1)
    return X


def prepare_train(df: pd.DataFrame):
    columns_not_included = ['backdrop_path', 'homepage', 'id', 'imdb_id', 'poster_path', 'production_countries',
                            'spoken_languages', 'status', 'video']
    textual_columns = ['original_title', 'overview', 'tagline', 'title', 'Keywords']
    target_column = 'revenue'

    X = df[df[target_column] >= 10000]
    y = X[target_column]
    X = X.drop(columns_not_included + textual_columns + [target_column], axis=1)

    columns_to_dict = ['genres', 'belongs_to_collection', 'cast',  'production_companies', 'crew']
    for col in columns_to_dict:
        X[col] = X[col].apply(lambda x: ast.literal_eval(str(x)) if type(x) == str else x)

    X['budget'] = X['budget'].apply(lambda x: x if x >= 2500 else np.nan)

    X["release_date"] = X["release_date"].astype("datetime64")
    X['release_date'] = X['release_date'].apply(lambda x: x.date().year) # take only year as int

    X['runtime'].replace({0: np.nan}, inplace=True)
    X['belongs_to_collection'] = X['belongs_to_collection'].apply(lambda x: [x] if type(x) in [dict] else x)

    # TODO place to change thresholds of #
    X, genres_dict = extract_and_threshold(X, 'genres', threshold=1)
    X, collections_dict = extract_and_threshold(X, 'belongs_to_collection', threshold=1)
    X, actors_dict = extract_and_threshold(X, 'cast', threshold=10)
    X, companies_dict = extract_and_threshold(X, 'production_companies', threshold=20)

    X['crew'] = X['crew'].apply(lambda x: extract_all_names_for_crew(x) if type(x) in [dict, list] else None)
    all_crew = [sublist for sublist in X['crew'].values if sublist]
    all_crew = [item for sublist in all_crew for item in sublist]
    all_crew = pd.Series(all_crew).value_counts()
    crew_to_keep = list(all_crew[all_crew >= 10].index)
    crew_to_keep = {category: i for i, category in enumerate(crew_to_keep)}
    X['crew'] = X['crew'].apply(lambda row: [] if row is None else row)
    X['crew'] = X['crew'].apply(lambda row: create_multi_hot(row, crew_to_keep))

    dicts = [genres_dict, collections_dict, actors_dict, companies_dict, crew_to_keep]
    prefixes_names = ['genre', 'collection', 'actor',  'company', 'crew']
    X = X.reset_index(drop=True)
    for col, dictionary, prefix in zip(columns_to_dict, dicts, prefixes_names):
        mat = np.vstack(list(X[col].values))
        columns_names = [prefix+'_'+i for i in dictionary.keys()]
        df2 = pd.DataFrame(mat, columns=columns_names)
        X = pd.concat([X, df2], axis=1)

    X = X.drop(columns_to_dict, axis=1)
    return X, y, dicts


def extract_all_names_for_crew(row):
    names = list()
    jobs = ['Executive Producer', 'Producer', 'Director']
    for x in row:
        if type(x) == dict and '<' not in str(x['name']) and x['job'] in jobs:
            names.append(x.get('name', '<no crew>'))
    return names[:20]


def extract_all_names(row):
    names = list()
    for x in row:
        if type(x) == dict and '<' not in str(x['name']):
            names.append(x.get('name', '<no cast>'))
    return names[:20]  # TODO possible parameter, relevant mainly to # of actors per movie


def extract_and_threshold(X, col_name, threshold):
    X[col_name] = X[col_name].apply(lambda x: extract_all_names(x) if type(x) in [dict, list] else None)
    all_categories = [sublist for sublist in X[col_name].values if sublist]
    all_categories = [item for sublist in all_categories for item in sublist]
    all_categories = pd.Series(all_categories).value_counts()
    categories_to_keep = list(all_categories[all_categories >= threshold].index)
    categories_to_keep = {category: i for i, category in enumerate(categories_to_keep)}
    X[col_name] = X[col_name].apply(lambda row: [] if row is None else row)
    X[col_name] = X[col_name].apply(lambda row: create_multi_hot(row, categories_to_keep))
    return X, categories_to_keep


def create_multi_hot(row, mapping_dict):
    multi_hot = [0] * len(mapping_dict)
    for element in row:
        if element in mapping_dict.keys():
            multi_hot[mapping_dict[element]] += 1
    return multi_hot


def load_data_for_exp11():
    """
    loads the train and test data ready for the baseline settings (features)
    :return: (X_train, y_train), (X_test, y_test). X is pd.DataFrame, y is pd.Series
    """
    train_df = pd.read_csv(TRAIN_PATH, sep='\t')
    test_df = pd.read_csv(TEST_PATH, sep='\t')
    X_train, y_train, dicts = prepare_train(train_df)

    y_test = None  # TODO decide how to define it for comp
    if 'revenue' in test_df.columns:  # TODO whether its test or comp
        y_test = test_df['revenue']
        test_df = test_df.drop(['revenue'], axis=1)
    X_test = prepare_test(test_df, dicts=dicts)

    return X_train, y_train, X_test, y_test
