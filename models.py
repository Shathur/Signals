from xgboost import XGBRegressor
import lightgbm as lgb


def run_model(train_data=None, val_data=None, model_type='xgb', save_to_drive=False, save_folder=None, cv_count=None):
    X_train, y_train = train_data
    X_val, y_val = val_data

    model = create_model(model_type=model_type)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)

    if save_to_drive:
        model.save_model(save_folder + 'model_{}.'.format(cv_count)+model_type)
    else:
        model.save_model('model_{}.'.format(cv_count)+model_type)

    return model


def create_model(model_type='xgb'):
    if model_type == 'lgb':
        params = {
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_leaves': 2 ** 5,
            'max_depth': 5,
            'colsample_bytree': 0.6,
            'device': "gpu",
        }
        model = lgb.LGBMRegressor(params=params)

    if model_type == 'xgb':
        model = XGBRegressor(max_depth=4, learning_rate=0.01, n_estimators=100, n_jobs=-1, colsample_bytree=0.1,
                             tree_method='gpu_hist', verbosity=0)  # tree_method='gpu_hist',

    return model
