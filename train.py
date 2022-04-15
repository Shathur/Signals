import pandas as pd
import utils
import models


def train_val(df, feature_names, target_name, pred_name, cv_split_data, date_col='date',
              tour_df=None, model_type='xgb', model_params=None, fit_params=None,
              save_to_drive='False', legacy_save=True, save_folder='None', visualize=True):
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
    :param legacy_save: True save with binary format, False save json, default True
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
        print("Training model on CV : {} with indices  train: {} to {}".format(cv_count, idx_cv[0][0], idx_cv[0][-1]))
        print('                                         val: {} to {}'.format(idx_cv[1][0], idx_cv[1][-1]))
        print('********************************************************************************************')

        train_tuple = [X_train, y_train]
        val_tuple = [X_val, y_val]

        model = models.run_model(train_data=train_tuple, val_data=val_tuple, model_type=model_type,
                                 model_params=model_params, fit_params=fit_params, save_to_drive=save_to_drive,
                                 legacy_save=legacy_save, save_folder=save_folder, cv_count=cv_count)

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

def train_CV(data_dir, last_friday, features_boundaries, model_name, target_name=TARGET_NAME, pred_name=PREDICTION_NAME,
             n_splits=4,
             submit=True, submit_diagnostics=False, submit_reverse=False, submit_diagnostics_reverse=False,
             model_name_reverse=None,
             models_save_folder='/content/mymodels/', upload_name='signal_upload'):
    # # preprocess data
    # prices_df, train_df, feature_names = prepare_dataset(data_dir, features_start)

    train_df = pd.read_parquet(data_dir)

    train_without_live = train_df[train_df['friday_date'] < last_friday]
    train_without_live.dropna(inplace=True)
    train_df = pd.concat([train_without_live, train_df[train_df['friday_date'] == last_friday]])

    feature_names = train_df.columns[
                    train_df.columns.str.find(features_boundaries[0]).argmax(): train_df.columns.str.find(
                        features_boundaries[1]).argmax() + 1].tolist()

    # split data
    cv_split_data = cv_split_creator(df=train_df, col='friday_date', n_splits=n_splits)

    # keep data for validation of the CV strategy
    tour_data = train_df.iloc[cv_split_data[0][1]]

    if os.path.isdir(models_save_folder):
        shutil.rmtree(models_save_folder)  # delete dir + models
    os.mkdir(models_save_folder)  # create dir from scratch

    metrics = train_val(df=train_df,
                        date_col='friday_date',
                        feature_names=feature_names,
                        target_name=TARGET_NAME,
                        pred_name=PREDICTION_NAME,
                        cv_split_data=cv_split_data,
                        tour_df=tour_data,

                        save_to_drive=True,
                        save_folder=models_save_folder,
                        visualize=True)

    # preds_total = metrics[2]
    # tour_preds_total = preds_total[2]

    # # ensemble the predictions
    # tour_data[PREDICTION_NAME] = np.mean(tour_preds_total, axis=0)
    # # calculate scores
    # tour_era_scores = tour_data.groupby(tour_data['date']).apply(lambda x: score(x, target_name, pred_name))
    # # for validation metrics
    # # run_analytics(tour_era_scores)

    validation_predictions = np.mean(metrics[2][2], axis=0)

    validation_sub = tour_data.copy()

    validation_sub['signal'] = validation_predictions

    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    last_friday = int(last_friday.strftime('%Y%m%d'))

    # live sub
    train_df.loc[train_df['friday_date'] == last_friday, 'data_type'] = 'live'

    live_sub = train_df.query('data_type == "live"').copy()

    live_sub['signal'] = get_predictions(df=live_sub[feature_names],
                                         num_models=4,
                                         folder_name='/content/mymodels/')

    # concat valid and test
    sub = pd.concat([validation_sub, live_sub], ignore_index=True)

    # select necessary columns
    sub = sub[['ticker', 'friday_date', 'data_type', 'signal']]

    # submit
    submit_signal(sub, public_id, secret_key, submit, submit_diagnostics, model_name, upload_name=upload_name)

    # submit reverse
    reverse_sub = sub
    reverse_sub['signal'] = reverse_sub.groupby('friday_date')['signal'].rank(pct=True, method="first", ascending=False)
    reverse_sub.reset_index(drop=True, inplace=True)
    submit_signal(reverse_sub, public_id, secret_key, submit_reverse, submit_diagnostics_reverse, model_name_reverse,
                  upload_name)

    # free memory
    gc.collect()
