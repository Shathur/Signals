import pandas as pd
import utils
import models


def train_val(df, feature_names, target_name, pred_name, cv_split_data, date_col='date',
              tour_df=None, model_type='xgb', model_params=None, save_to_drive='False',
              save_folder='None', visualize=True):
    """

    :param date_col: time column
    :param df: your dataframe to be split and trained on
    :param feature_names: features to be trained on
    :param target_name: our target
    :param pred_name: our predictions
    :param cv_split_data: the iterator that splits our data
    :param tour_df: the validation data we will monitor at the end
    :param model_type: 'xgb' or 'lgb'
    :param model_params: custom parameters for the Regressors. If None load default
    :param save_to_drive: True save False don't save
    :param save_folder: path destination of models
    :param visualize: boolean, default True
    :return: feature importances, hit rates per split, list with train, oof and validation predictions
    """
    train_preds_total = []
    val_preds_total = []
    tour_preds_total = []

    feat_importances_total = []

    diagnostics_per_split = []

    for cv_count, idx_cv in enumerate(cv_split_data):
        train_data = df.iloc[idx_cv[0]]
        val_data = df.iloc[idx_cv[1]]

        X_train = train_data[feature_names]
        y_train = train_data[target_name]
        X_val = val_data[feature_names]
        y_val = val_data[target_name]

        print('********************************************************************************************')
        print("Training model on CV : {} with indices  train: {} to {}".format(cv_count, idx_cv[0][0], idx_cv[0][1]))
        print('                                         val: {} to {}'.format(idx_cv[1][0], idx_cv[1][1]))
        print('********************************************************************************************')

        train_tuple = [X_train, y_train]
        val_tuple = [X_val, y_val]

        model = models.run_model(train_data=train_tuple, val_data=val_tuple, model_type=model_type,
                                 model_params=model_params, save_to_drive=save_to_drive, save_folder=save_folder,
                                 cv_count=cv_count)

        if visualize:
            utils.plot_feature_importances(feature_names, model)
        feat_importances = model.feature_importances_
        feat_importances_dict = dict(zip(feature_names, feat_importances))
        feat_importances_total.append(feat_importances_dict)

        train_preds = model.predict(train_data[feature_names])
        val_preds = model.predict(val_data[feature_names])

        train_preds_total.append(train_preds)
        val_preds_total.append(val_preds)

        train_data[pred_name] = train_preds
        val_data[pred_name] = val_preds

        if visualize:
            # show prediction distribution, most should be around the center
            val_data[pred_name].hist(bins=30)

        # spearman scores by era
        train_era_scores = train_data.groupby(train_data[date_col]).apply(lambda x: utils.score(x, target_name, pred_name))
        val_era_scores = val_data.groupby(val_data[date_col]).apply(lambda x: utils.score(x, target_name, pred_name))

        # test scores, out of sample
        hit_train = utils.run_analytics(train_era_scores)
        hit_val = utils.run_analytics(val_era_scores, plot_figures=visualize)

        # keep everything in a neat dataframe
        train_start, train_end = utils.start_end_date(train_data, date_col)
        val_start, val_end = utils.start_end_date(val_data, date_col)

        if tour_df is not None:
            tour_preds = model.predict(tour_df[feature_names])
            tour_preds_total.append(tour_preds)
            tour_df[pred_name] = tour_preds
            tour_era_scores = tour_df.groupby(tour_df[date_col]).apply(lambda x: utils.score(x, target_name, pred_name))
            hit_tour = utils.run_analytics(tour_era_scores)

        dic = {'train_start': train_start,
               'train_end': train_end,
               'train_hit': hit_train,
               'val_start': val_start,
               'val_end': val_end,
               'val_hit': hit_val}

        if tour_df is not None:
            dic.update({'tour_hit': hit_tour})

        diagnostics_per_split.append(dic)

    # keep everything in a tidy df
    feat_importances_df = pd.DataFrame(feat_importances_total)
    diagnostics_per_split_df = pd.DataFrame(diagnostics_per_split)

    preds_total = [train_preds_total,
                   val_preds_total,
                   tour_preds_total]

    return feat_importances_df, diagnostics_per_split_df, preds_total
