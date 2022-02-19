from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb


def run_model(train_data=None, val_data=None, model_type='xgb', task_type='regression', model_params=None,
              fit_params=None, save_to_drive=False, save_folder=None, cv_count=None):
    X_train, y_train = train_data
    X_val, y_val = val_data

    model = create_model(model_type=model_type, task_type=task_type, model_params=model_params)

    if fit_params is None:
        fit_params = {
            'early_stopping_rounds': 10,
            'verbose': False
        }
    else:
        pass

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)

    if save_to_drive:
        model.save_model(save_folder + 'model_{}.'.format(cv_count)+model_type)
    else:
        model.save_model('model_{}.'.format(cv_count)+model_type)

    return model


def create_model(model_type='xgb', task_type='regression', model_params=None):
    if model_params is None:
        model_params = get_default_params(model_type=model_type)
    else:
        pass
    if model_type == 'lgb':
        model = lgb.LGBMRegressor()
        model.set_params(**model_params)

    if model_type == 'xgb':
        model = XGBRegressor()
        model.set_params(**model_params)

    if task_type == 'classification':
        model = XGBClassifier()
        model.set_params(**model_params)

    return model


def get_default_params(model_type='xgb'):
    if model_type == 'lgb':
        params = {
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'num_leaves': 2 ** 5,
            'max_depth': 5,
            'colsample_bytree': 0.6,
            'device': "gpu",
        }
    elif model_type == 'xgb':
        params = {
            'max_depth': 5,
            'learning_rate': 0.01,
            'n_estimators': 1000,
            'n_jobs': -1,
            'colsample_bytree': 0.6,
            'tree_method': 'gpu_hist',
            'verbosity': 0
        }

    return params
