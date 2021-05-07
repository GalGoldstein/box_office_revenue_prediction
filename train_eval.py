import preprocess_data
import numpy as np
import models


def rmsle(preds, true):
    """
    Compute the Root Mean Squared Log Error for predictions `preds` and targets `true`

    :param preds: numpy array containing predictions with shape (n_samples, n_targets)
    :param true: numpy array containing targets with shape (n_samples, n_targets)
    :return: RMSLE score
    """
    return np.sqrt(np.square(np.log(true + 1) - np.log(preds + 1)).mean())


def train_eval_baseline():
    (X_train, y_train), (X_test, y_test) = preprocess_data.load_data_for_baseline_ml()
    categorical_features = ['belongs_to_collection', 'genres', 'original_language', 'production_companies',
                            'cast_1', 'cast_2', 'cast_3', 'cast_4', 'cast_5',  # cast features
                            'Executive Producer', 'Producer', 'Director', 'Screenplay', 'Author']  # crew features`

    model = models.catboost_model(categorical_features)

    # -------------------------------------------------
    # filter train data TODO Exp4
    # X_train[y_train.name] = y_train
    # X_train = X_train[X_train['revenue'] > 10000]
    # y_train = X_train['revenue']
    # X_train = X_train.drop(['revenue'], axis=1)
    # -------------------------------------------------

    # -------------------------------------------------
    print('a')
    # -------------------------------------------------

    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    test_rmsle = rmsle(test_preds, y_test)
    print(f'-------- Test RMSLE -------- {test_rmsle}')


if __name__ == '__main__':
    train_eval_baseline()
