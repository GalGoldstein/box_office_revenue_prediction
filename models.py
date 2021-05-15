import catboost as cb
import lightgbm as lgb


def catboost_model(categorical_features):
    model = cb.CatBoostRegressor(cat_features=categorical_features)
    return model


def lgbm_model_and_training(lgb_data):
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'rmsle',
              'max_depth': -1,
              'num_leaves': 170,
              'cat_smooth': 10.0,  # 10.0 is default
              'learning_rate': 0.1,
              'verbose': 1}
    model = lgb.train(params, lgb_data)
    return model, params
