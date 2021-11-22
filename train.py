import pandas as pd
import utils
import models


def train_val(train_df, feature_names, target_name, pred_name, cv_split_data,
              model_type='xgb', save_to_drive='False', save_folder='None'):
    cv_count = 0

    train_preds_total = []
    val_preds_total = []

    feat_importances_total = []

    diagnostics_per_split = []

    for cv_count, idx_cv in enumerate(cv_split_data):
        train_data = train_df.iloc[idx_cv[0]]
        val_data = train_df.iloc[idx_cv[1]]

        X_train = train_data[feature_names]
        y_train = train_data[target_name]
        X_val = val_data[feature_names]
        y_val = val_data[target_name]

        print('********************************************************************************************')
        print("Training model on CV : {} with indixes : {}".format(cv_count, idx_cv))
        print('********************************************************************************************')

        train_tuple = [X_train, y_train]
        val_tuple = [X_val, y_val]

        model = models.run_model(train_data=train_tuple, val_data = val_tuple, model_type=model_type,
                                 save_to_drive=save_to_drive, save_folder = save_folder, cv_count=cv_count)

        cv_count += 1

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

        # show prediction distribution, most should around the center
        val_data[pred_name].hist(bins=30)

        # spearman scores by era
        train_era_scores = train_data.groupby(train_data['date']).apply(lambda x: utils.score(x, target_name, pred_name))
        val_era_scores = val_data.groupby(val_data['date']).apply(lambda x: utils.score(x, target_name, pred_name))

        # test scores, out of sample
        hit_train = utils.run_analytics(train_era_scores)
        hit_val = utils.run_analytics(val_era_scores)

        # keep everything in a neat dataframe
        train_start, train_end = utils.start_end_date(train_data)
        val_start, val_end = utils.start_end_date(val_data)

        dic = {'train_start': train_start,
               'train_end': train_end,
               'train_hit': hit_train,
               'val_start': val_start,
               'val_end': val_end,
               'val_hit': hit_val}

        diagnostics_per_split.append(dic)

    # keep everything in a tidy df
    feat_importances_df = pd.DataFrame(feat_importances_total)
    diagnostics_per_split_df = pd.DataFrame(diagnostics_per_split)

    return feat_importances_df, diagnostics_per_split_df
