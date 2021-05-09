import preprocess_data
import numpy as np
import models
import lightgbm as lgb


def rmsle(preds, true):
    """
    Compute the Root Mean Squared Log Error for predictions `preds` and targets `true`

    :param preds: predictions
    :param true: targets
    :return: RMSLE score
    """
    return np.sqrt(np.square(np.log(true + 1) - np.log(preds + 1)).mean())


def train_eval_baseline(model: str, scale_method: str):
    (X_train, y_train), (X_test, y_test) = preprocess_data.load_data_for_baseline_ml()
    categorical_features = ['belongs_to_collection', 'genres', 'original_language', 'production_companies',
                            'cast_1', 'cast_2', 'cast_3', 'cast_4', 'cast_5',  # cast features
                            'Executive Producer', 'Producer', 'Director', 'Screenplay', 'Author']  # crew features`

    if scale_method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    if scale_method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    if scale_method == 'log':
        y_train = np.log1p(y_train)

    if model == 'catboost':
        model = models.catboost_model(categorical_features)

        # -------------------------------------------------
        # filter train data TODO Exp4
        # X_train[y_train.name] = y_train
        # X_train = X_train[X_train['revenue'] > 10000]
        # y_train = X_train['revenue']
        # X_train = X_train.drop(['revenue'], axis=1)
        # -------------------------------------------------

        model.fit(X_train, y_train)
    if model == 'lgbm':
        for col in categorical_features:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        lgb_data = lgb.Dataset(X_train, y_train)
        model, params = models.lgbm_model_and_training(lgb_data)
        print(f"------------- LGBM params ------------- \n {params}")

    test_preds = model.predict(X_test)
    if scale_method in ['minmax', 'standard']:
        test_preds = scaler.inverse_transform(test_preds.reshape(-1, 1)).reshape(-1)

    if scale_method == 'log':
        test_preds = np.expm1(test_preds)

    test_rmsle = rmsle(test_preds, y_test)
    print(f'-------- Test RMSLE -------- {test_rmsle}')


if __name__ == '__main__':
    train_eval_baseline(model='catboost', scale_method='log')
