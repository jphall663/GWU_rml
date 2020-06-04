import h2o
import numpy as np
import pandas as pd

"""

Copyright 2020 - Patrick Hall (jphall@gwu.edu) 
Copyright 2020 - Patrick Hall (phall@h2o.ai) and the H2O.ai team

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

"""

"""

All aspects of rmltk are based on public ideas, e.g.: 

- https://bit.ly/2KmUN2J
- https://bit.ly/3bmvVEf

"""

# TODO: unit tests
# TODO: model documentation
# TODO: CV metrics for all measures, not just accuracy
# TODO: Multi-threading for cv rank?


def cv_model_rank(valid, seed_, model_name_list, nfolds=5):

    """ Rough implementation of CV model ranking used in 2004 KDD Cup:
    https://dl.acm.org/doi/pdf/10.1145/1046456.1046470. Evaluates model
    ranks across random folds based on multiple measures.

    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param model_name_list: A list of strings in which each token is the name
                            of the Python reference to an H2O model.
    :param nfolds: Number of folds over which to evaluate model rankings.

    :return: A Pandas frame with model ranking information.
    """

    # must be metrics supported by h2o
    # assumes binary classification classification
    metric_name_list = ['mcc', 'F1', 'accuracy', 'logloss', 'auc']

    # copy original frame
    # create reproducible folds
    temp_df = valid.copy(deep=True)
    np.random.seed(seed=seed_)
    temp_df['fold'] = np.random.randint(low=0, high=nfolds, size=temp_df.shape[0])

    # initialize the returned eval_frame
    # columns for rank added later
    columns_ = ['Fold', 'Metric']
    columns_ += [model + ' Value' for model in model_name_list]
    eval_frame = pd.DataFrame(columns=columns_)

    # loop counter
    i = 0

    # loop through folds and metrics
    for fold in sorted(temp_df['fold'].unique()):
        for metric in sorted(metric_name_list):

            # necessary for adding more than one value per loop iteration
            # and appending those to eval_frame conveniently
            val_dict = {}

            # dynamically generate and run code statements
            # to calculate metrics for each fold and model
            for model in sorted(model_name_list):
                code = 'h2o.get_model("%s").model_performance(h2o.H2OFrame(temp_df[temp_df["fold"] == %d])).%s()' \
                       % (model, fold, metric)
                key_ = model + ' Value'
                val_ = eval(code)

                # some h2o metrics are returned as a list
                # this may make an assumption about binary classification?
                if isinstance(val_, list):
                    val_ = val_[0][1]
                val_dict[key_] = val_

            # create columns to store rankings
            rank_list = list(val_dict.keys())

            # add fold label and metric name into val_dict
            # with multiple model names and metric values generated above
            # append all to eval_frame
            val_dict.update({
                'Fold': fold,
                'Metric': metric})
            eval_frame = eval_frame.append(val_dict, ignore_index=True)

            # add rankings into the same row
            # conditional on direction of metric improvement
            for val_ in sorted(rank_list):
                if eval_frame.loc[i, 'Metric'] == 'logloss':
                    eval_frame.loc[i, val_.replace(' Value', ' Rank')] = eval_frame.loc[i, rank_list].rank()[val_]
                else:
                    eval_frame.loc[i, val_.replace(' Value', ' Rank')] = \
                        eval_frame.loc[i, rank_list].rank(ascending=False)[val_]

            i += 1

    del temp_df

    return eval_frame


def cv_model_rank_select(valid, seed_, train_results, model_prefix,
                         compare_model_ids, nfolds=5):

    """ Performs CV ranking for models in model_list, as compared
        to other models in model_name_list and automatically
        selects highest ranking model across the model_list.

    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param train_results: Dict created by gbm_forward_select_train
                          containing a list of models, a list of
                          global coefficients, and a list of local
                          coefficients.
    :param model_prefix: String prefix for generated model_id's.
    :param compare_model_ids: A list of H2O model_ids.
    :param nfolds: Number of folds over which to evaluate model rankings.

    :return: Best model from model_list, it's associated
             coefficients from coef_list, and the CV rank eval_frame
             for the best model.
    """

    best_idx = 0
    rank = len(compare_model_ids) + 1
    best_model_frame = None

    for i in range(0, len(train_results['MODELS'])):

        # assign model_ids correctly
        # so models can be accessed by model_id
        # in cv_model_rank
        model_id = model_prefix + str(i+1)
        train_results['MODELS'][i].model_id = model_id
        model_name_list = compare_model_ids + [model_id]

        # perform CV rank eval for
        # current model in model list vs. all compare models
        eval_frame = cv_model_rank(valid, seed_, model_name_list, nfolds=nfolds)

        # cache CV rank of current model
        title_model_col = model_name_list[-1] + ' Rank'
        new_rank = eval_frame[title_model_col].mean()

        # determine if this model outranks
        # previous best models
        if new_rank < rank:
            best_idx = i
            best_model_frame = eval_frame
            print('Evaluated model %i/%i with rank: %.2f* ...' % (i + 1, len(train_results['MODELS']),
                                                                  new_rank))
            rank = new_rank
        else:
            print('Evaluated model %i/%i with rank: %.2f ...' % (i + 1, len(train_results['MODELS']),
                                                                 new_rank))

    # select model and coefficients
    best_model = train_results['MODELS'][best_idx]
    best_shap = train_results['LOCAL_COEFS'][best_idx]
    best_coefs = train_results['GLOBAL_COEFS'][best_idx]

    print('Done.')

    # return best model, it's associated coefficients
    # and it's CV ranking frame
    return {'BEST_MODEL': best_model,
            'BEST_LOCAL_COEFS': best_shap,
            'BEST_GLOBAL_COEFS': best_coefs,
            'METRICS': best_model_frame}


def get_prauc(frame, y, yhat, pos=1, neg=0, res=0.01):

    """ Calculates precision, recall, and f1 for a pandas dataframe of y and yhat values.

    Args:
        frame: Pandas dataframe of actual (y) and predicted (yhat) values.
        y: Name of actual value column.
        yhat: Name of predicted value column.
        pos: Primary target value, default 1.
        neg: Secondary target value, default 0.
        res: Resolution by which to loop through cutoffs, default 0.01.

    Returns:
        Pandas dataframe of precision, recall, and f1 values.
    """

    frame_ = frame.copy(deep=True)  # don't destroy original data
    dname = 'd_' + str(y)  # column for predicted decisions
    eps = 1e-20  # for safe numerical operations

    # init p-r roc frame
    prauc_frame = pd.DataFrame(columns=['cutoff', 'recall', 'precision', 'f1'])

    # loop through cutoffs to create p-r roc frame
    for cutoff in np.arange(0, 1 + res, res):
        # binarize decision to create confusion matrix values
        frame_[dname] = np.where(frame_[yhat] > cutoff, 1, 0)

        # calculate confusion matrix values
        tp = frame_[(frame_[dname] == pos) & (frame_[y] == pos)].shape[0]
        fp = frame_[(frame_[dname] == pos) & (frame_[y] == neg)].shape[0]
        fn = frame_[(frame_[dname] == neg) & (frame_[y] == pos)].shape[0]

        # calculate precision, recall, and f1
        recall = (tp + eps) / ((tp + fn) + eps)
        precision = (tp + eps) / ((tp + fp) + eps)
        f1 = 2 / ((1 / (recall + eps)) + (1 / (precision + eps)))

        # add new values to frame
        prauc_frame = prauc_frame.append({'cutoff': cutoff,
                                          'recall': recall,
                                          'precision': precision,
                                          'f1': f1},
                                         ignore_index=True)

    # housekeeping
    del frame_

    return prauc_frame


def get_youdens_j(frame, y, yhat, pos=1, neg=0, res=0.01):

    """ Calculates TPR, TNR, and Youden's J for a Pandas DataFrame of actual (_y_name) and predicted (_yhat_name) values
        to select best cutoff for AUC-optimized classifier.

        :param frame: Pandas DataFrame of actual (_y_name) and predicted (_yhat_name) values.
        :param y: Name of actual value column.
        :param yhat: Name of predicted value column.
        :param pos: Primary target value, default 1.
        :param neg: Secondary target value, default 0.
        :param res: Resolution by which to loop through cutoffs, default 0.01.
        :return: Pandas DataFrame of sensitivity, specificity, and Youden's J values.

    """

    frame_ = frame.copy(deep=True)  # don't destroy original data
    dname = 'd_' + str(y)  # column for predicted decisions
    eps = 1e-20  # for safe numerical operations

    # init j_frame
    j_frame = pd.DataFrame(columns=['cutoff', 'TPR', 'TNR', 'J'])

    # loop through cutoffs to create j_frame
    for cutoff in np.arange(0, 1 + res, res):

        # binarize decision to create confusion matrix values
        frame_[dname] = np.where(frame_[yhat] > cutoff, 1, 0)

        # calculate confusion matrix values
        tp = frame_[(frame_[dname] == pos) & (frame_[y] == pos)].shape[0]
        fp = frame_[(frame_[dname] == pos) & (frame_[y] == neg)].shape[0]
        tn = frame_[(frame_[dname] == neg) & (frame_[y] == neg)].shape[0]
        fn = frame_[(frame_[dname] == neg) & (frame_[y] == pos)].shape[0]

        # calculate precision, recall, and Youden's J
        tpr = (tp + eps) / ((tp + fn) + eps)
        tnr = (tn + eps) / ((tn + fp) + eps)
        fnr = 1 - tnr
        j = tpr + tnr - 1

        # add new values to frame
        j_frame = j_frame.append({'cutoff': cutoff,
                                  'TPR': tpr,
                                  'TNR': tnr,
                                  'FNR': fnr,
                                  'J': j},
                                 ignore_index=True)

    # housekeeping
    del frame_

    return j_frame


def get_confusion_matrix(valid, y_name, yhat_name, by=None, level=None, cutoff=0.5):

    """ Creates confusion matrix from pandas DataFrame of y and yhat values, can be sliced
        by a variable and level.

        :param valid: Validation DataFrame of actual (y) and predicted (yhat) values.
        :param y_name: Name of actual value column.
        :param yhat_name: Name of predicted value column.
        :param by: By variable to slice frame before creating confusion matrix, default None.
        :param level: Value of by variable to slice frame before creating confusion matrix, default None.
        :param cutoff: Cutoff threshold for confusion matrix, default 0.5.
        :return: Confusion matrix as pandas DataFrame.

    """

    # determine levels of target (y) variable
    # sort for consistency
    level_list = list(valid[y_name].unique())
    level_list.sort(reverse=True)

    # init confusion matrix
    cm_frame = pd.DataFrame(columns=['actual: ' + str(i) for i in level_list],
                            index=['predicted: ' + str(i) for i in level_list])

    # don't destroy original data
    frame_ = valid.copy(deep=True)

    # convert numeric predictions to binary decisions using cutoff
    dname = 'd_' + str(y_name)
    frame_[dname] = np.where(frame_[yhat_name] > cutoff, 1, 0)

    # slice frame
    if (by is not None) & (level is not None):
        frame_ = frame_[valid[by] == level]

    # calculate size of each confusion matrix value
    for i, lev_i in enumerate(level_list):
        for j, lev_j in enumerate(level_list):
            cm_frame.iat[j, i] = frame_[(frame_[y_name] == lev_i) & (frame_[dname] == lev_j)].shape[0]
            # i, j vs. j, i nasty little bug ... updated 8/30/19

    return cm_frame

