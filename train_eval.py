"""
This script contains three function to train and evaluate in three different techniques:
1. Our baseline fata representation with catboost / lgbm models
2. pycaret models training to find the best models that suit to our baseline data representation
3. pycaret models training with a multi-hot data representation

also in this script is `rmsle` function for evaluating our models
"""

import preprocess_data
import numpy as np
import models
import lightgbm as lgb
import pandas as pd
from pycaret.regression import setup, compare_models, finalize_model, predict_model, save_model
from multi_hot_transformations import load_data_for_exp11


def rmsle(preds, true):
    """
    Compute the Root Mean Squared Log Error for predictions `preds` and targets `true`

    :param preds: predictions
    :param true: targets
    :return: RMSLE score
    """
    return np.sqrt(np.square(np.log(true + 1) - np.log(preds + 1)).mean())


def train_eval_baseline(model_name: str, target_scale_method: str):
    (X_train, y_train), (X_test, y_test) = preprocess_data.load_data_for_ml()
    categorical_features = ['belongs_to_collection', 'original_language',
                            'production_company_1', 'production_company_2', 'production_company_3',
                            'genre_1', 'genre_2', 'genre_3',
                            'cast_1', 'cast_2', 'cast_3', 'cast_4', 'cast_5',  # cast features
                            'Executive Producer', 'Producer', 'Director', 'Screenplay', 'Author']  # crew features`

    if target_scale_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    if target_scale_method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    if target_scale_method == 'log':
        y_train = np.log1p(y_train)

    if model_name == 'catboost':
        model = models.catboost_model(categorical_features)
        model.set_params(**dict(iterations=800))
        model.fit(X_train, y_train)

        # -------------------------------------------------
        # hyperparameters
        # grid = {
        #     'learning_rate': np.arange(0.03, 0.1, 0.01),
        #     'depth': np.arange(5, 20, 1),
        #     'l2_leaf_reg': np.arange(0.5, 10, 0.5),
        #     'iterations': np.arange(700, 1200, 50),
        #     'one_hot_max_size': np.arange(10, 50, 10),
        #     'bagging_temperature': np.arange(1, 100, 1),
        #     'rsm': np.arange(0.3, 1, 0.05)
        # }
        # randomized_search_result = model.randomized_search(grid,
        #                                                    X=X_train,
        #                                                    y=y_train,
        #                                                    n_iter=50)
        # # -------------------------------------------------
        # model = cb.CatBoostRegressor(cat_features=categorical_features)
        # model.set_params(**randomized_search_result['params'])
        # model.fit(X_train, y_train)

        print(f'-------------- all catboost parmas -------------- \n{model.get_all_params()}')

    if model_name == 'lgbm':
        for col in categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        lgb_data = lgb.Dataset(X_train, y_train)
        model, params = models.lgbm_model_and_training(lgb_data)
        print(f"------------- LGBM params ------------- \n {params}")

    test_preds = model.predict(X_test)

    if target_scale_method in ['minmax', 'standard']:
        test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(-1)

    if target_scale_method == 'log':
        test_preds = np.expm1(test_preds)

    test_rmsle = rmsle(test_preds, y_test)
    print(f'-------- Test RMSLE -------- {test_rmsle}')


def train_eval_pycaret(target_scale_method: str):
    (X_train, y_train), (X_test, y_test), num_categoricals = \
        preprocess_data.load_data_transform2DFNumpy(scale_popularity=True)
    # prepare train data
    df_train = pd.DataFrame(X_train)
    if target_scale_method == 'log':
        df_train['revenue'] = np.log1p(y_train)
    else:
        df_train['revenue'] = y_train
    df_train.columns = df_train.columns.astype(str)

    # prepare test data
    df_test = pd.DataFrame(X_test)
    if target_scale_method == 'log':
        df_test['revenue'] = np.log1p(y_test)
    else:
        df_test['revenue'] = y_test
    df_test.columns = df_test.columns.astype(str)

    categorical_features = [str(i) for i in range(num_categoricals)]

    # pycaret
    regressor = setup(data=df_train,
                      target='revenue',  # label column
                      categorical_features=categorical_features,  # will be encoded to one-hot
                      numeric_imputation='median',  # each numrical nan replaced with the column median
                      silent=True,  # True tells pycaret to not ask for approval of columns types
                      session_id=555)  # like random_seed

    # find best model
    # best = compare_models(include=['xgboost', 'rf', 'en', 'lar', 'llar', 'mlp',
    #                                'gbr', 'lightgbm', 'catboost', 'svm', 'et'],
    #                       sort='RMSLE')
    # final = finalize_model(best)

    # or choose specific model
    best = compare_models(include=['catboost'], sort='RMSLE')
    final = finalize_model(best)

    # save_model(final, 'catboost_model')  # TODO

    test_preds = predict_model(final, data=df_test)['Label']
    if target_scale_method == 'log':
        test_preds = np.expm1(test_preds)

    test_rmsle = rmsle(test_preds, y_test)
    print(f'-------- Test RMSLE -------- {test_rmsle}')


def train_eval_multi_hot(target_scale_method: str):
    X_train, y_train, X_test, y_test = load_data_for_exp11()
    X_train['revenue'] = np.log1p(y_train) if target_scale_method == 'log' else y_train
    X_test['revenue'] = np.log1p(y_test) if target_scale_method == 'log' else y_test

    # remove nan
    X_train['budget'].replace({np.nan: np.mean(X_train['budget'])}, inplace=True)
    X_test['budget'].replace({np.nan: np.mean(X_train['budget'])}, inplace=True)

    X_train['runtime'].replace({np.nan: np.mean(X_train['runtime'])}, inplace=True)
    X_test['runtime'].replace({np.nan: np.mean(X_train['runtime'])}, inplace=True)

    categorical_features = ['original_language']

    # pycaret
    regressor = setup(data=X_train,
                      target='revenue',
                      categorical_features=categorical_features,
                      silent=True)

    # find best model
    # best = compare_models(include=['xgboost', 'rf', 'en', 'lar', 'llar', 'mlp',
    #                                'gbr', 'lightgbm', 'catboost', 'svm', 'et'],
    #                       sort='RMSLE')
    # final = finalize_model(best)

    # or choose specific model
    best = compare_models(include=['lightgbm'], sort='RMSLE')
    final = finalize_model(best)
    test_preds = predict_model(final, data=X_test)['Label']
    test_preds = np.expm1(test_preds) if target_scale_method == 'log' else test_preds

    test_rmsle = rmsle(test_preds, y_test)
    print(f'-------- Test RMSLE -------- {test_rmsle}')


if __name__ == '__main__':
    # train_eval_baseline(model_name='catboost', target_scale_method='log')  # reproduce 1.7678 test rmsle
    train_eval_pycaret(target_scale_method='log')
    # train_eval_multi_hot(target_scale_method='log')
