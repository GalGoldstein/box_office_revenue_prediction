import catboost as cb


def catboost_model(categorical_features):
    model = cb.CatBoostRegressor(cat_features=categorical_features)
    return model
