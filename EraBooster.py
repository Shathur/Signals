import pandas as pd

from Signals.predictions import get_predictions_per_era
from Signals.utils import get_era_idx, spearman

class EraBooster:
    """
    Class that helps us train iteratively on the worst performing eras
    
    ::param model: model that will be used for training
    ::param date_col: our time period, can be 'era', 'date', 'Moon' etc
    ::param prediction_col: our prediction column
    ::param target_col: our target column
    ::param added_trees: number of trees to be added in our booster in each iteration
    ::param quantile: percentage of worst performing eras to operate on
    
    ::param iteration: tracks iterations
    ::param era_diff_test_total_lst: list that keeps the difference between
        the era scores of the test dataframe
        in a neat dictionary with columns:
            -- iteration
            -- date_col
            -- correlation_changes
    ::param worst_eras_total: list with a dictionary that tracks the performance of
        each worst performing time period
    """

    def __init__(
        self,
        model,
        date_col,
        prediction_col,
        target_col,
        added_trees,
        quantile,
    ):
        self.model = model
        self.date_col = date_col
        self.prediction_col = prediction_col
        self.target_col = target_col
        self.added_trees = added_trees
        self.quantile = quantile
        
        self.iteration = 0
        self.era_diff_test_total_lst = []
        self.worst_eras_total = []

    def train_first(
        self,
        train_data,
        test_data,
    ):
        """
        Train model for the first time
        
        ::param train_data: train dictionary with 'X' and 'y' parts
        ::param test_data: test dictionary with 'X' and 'y' parts
        """
        print('Training for the first time ...')
        self.model.fit(
            train_data['X'],
            train_data['y'],
            eval_set=test_data,
    )

    def train_once(self, iter_count):
        print(f"Adding {added_trees} trees and training on worst eras ... Iter: {iter_count}")
        self.model.n_estimators += trees_per_step
        booster = self.model.get_booster()
        self.model.fit(
            
    )

    def return_predictions(self, train_df, test_df, validation_df):
        """
        return the predictions as a list of np.arrays for 
        train, test and validation sets
        """
        train_indices = get_era_idx(train_df,self.date_col)
        preds_train = get_predictions_per_era(
            df=train_df,
            num_models=1,
            era_idx=train_indices,
        )
        test_indices = get_era_idx(test_df,self.date_col)
        preds_test = get_predictions_per_era(
            df=test_df,
            num_models=1,
            era_idx=test_indices,
        )
        validation_indices = get_era_idx(validation_df,self.date_col)
        preds_validation = get_predictions_per_era(
            df=validation_df,
            num_models=1,
            era_idx=validation_indices,
        )
        return preds_train, preds_test, preds_validation

    def return_era_scores(self, train_df, test_df):
        """
        return the spearman correlations between our predictions and the 
        targets for each era, for the train and test dataset
        """
        era_scores_train = pd.Series(index=train_df[self.date_col].unique())
        for era in era_scores_train.index:
            era_df = train_df[train_df[self.date_col] == era]
            era_scores_train[era] = spearman(era_df[self.prediction_col], era_df[self.target_col]).correlation
        era_scores_test = pd.Series(index=test_df[self.date_col].unique())
        for era in era_scores_test.index:
            era_df = test_df[test_df[self.date_col] == era]
            era_scores_test[era] = spearman(era_df[self.prediction_col], era_df[self.target_col]).correlation
        return era_scores_train, era_scores_test

    def era_diff(self, test_df, era_scores_test, era_scores_test_prior):
        """
        updates the class variable era_diff_test_total which holds
        the difference between the era scores of the test dataframe
        in a neat dictionary with columns:
            -- iteration
            -- date_col
            -- correlation_changes
        It needs the output of return_era scores as input
        """
        era_diff_test = pd.Series(index=test_df[self.date_col].unique())
        if self.iteration > 0:
            for era in era_scores_test.index:
                era_diff_test[era] = era_scores_test[era] - era_scores_test_prior[era]
                # dic = {'iteration': 'it' + str(self.iteration), date_col: '_' + era[3:], 'correlation_changes': era_diff_test[era]}
                dic = {
                    'iteration': f"it{str(self.iteration)}",
                    self.date_col: f"_{int(era)}",
                    'correlation_changes': {era_diff_test[era]},
                }
                self.era_diff_test_total_lst.append(dic)

    def worst_scores(self, era_scores_train):
        """
        calculates the worst eras for this run and
        and updates the list of worst_eras_total
        """
        era_scores_train = era_scores_train.sort_values()
        worst_eras = era_scores_train[era_scores_train <= era_scores_train.quantile(self.quantile)].index.tolist()
        self.worst_eras_total.append(worst_eras)
