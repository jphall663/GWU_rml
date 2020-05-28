# Trains a simple NN on the Friedman data

# Python imports

import logging
from numpy.random import seed
import numpy as np
import pandas as pd
import os
import sys
import time

import h2o
from rmltk import model

# global training constants
EPOCHS = 20
PATIENCE = 5
SEED = 33333

# global system constants
PREFIX = 'friedman-siam-gbm-shap-all'

# start timer
tic = time.time()

# time stamp
time_stamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(tic))

# init logger
logger = logging.getLogger(__file__)

# init h2o
h2o.init(nthreads=4)


def get_confusion_matrix(frame_, y_name_, yhat_name_, by=None, level=None, cutoff=0.5, logger_=logger):

    """ Creates confusion matrix from pandas dataframe of y and yhat values, can be sliced
        by a variable and level.

    Args:
        frame_: Pandas dataframe of actual (y) and predicted (yhat) values.
        y_name_: Name of actual value column.
        yhat_name_: Name of predicted value column.
        by: By variable to slice frame before creating confusion matrix, default None.
        level: Value of by variable to slice frame before creating confusion matrix, default None.
        cutoff: Cutoff threshold for confusion matrix, default 0.5.
        logger_:

    Returns:
        Confusion matrix as pandas dataframe.
    """

    # determine levels of target (y) variable
    # sort for consistency
    level_list = list(frame_[y_name_].unique())
    level_list.sort(reverse=True)

    # init confusion matrix
    cm_frame = pd.DataFrame(columns=['actual: ' + str(i) for i in level_list],
                            index=['predicted: ' + str(i) for i in level_list])

    # don't destroy original data
    temp_df = frame_.copy(deep=True)

    # convert numeric predictions to binary decisions using cutoff
    dname = 'd_' + str(y_name_)
    temp_df[dname] = np.where(temp_df[yhat_name_] > cutoff, 1, 0)

    # slice frame
    if (by is not None) & (level is not None):
        temp_df = temp_df[frame_[by] == level]

    # calculate size of each confusion matrix value
    for i, lev_i in enumerate(level_list):
        for j, lev_j in enumerate(level_list):
            cm_frame.iat[j, i] = temp_df[(temp_df[y_name_] == lev_i) & (temp_df[dname] == lev_j)].shape[0]
            # i, j vs. j, i nasty little bug ... updated 8/30/19

    # output results
    if by is None:
        logger_.info('Confusion matrix at threshold = %.8f:' % cutoff)
    else:
        logger_.info('Confusion matrix by %s = %s at threshold = %.8f' % (by, level, cutoff))

    return cm_frame


def get_air(cm_dict_, control_level_, protected_level_):

    """ Calculates the adverse impact ratio as a quotient between protected and
        reference group acceptance rates: protected_prop/reference_prop.
        Prints intermediate values. Tightly coupled to cm_dict.
        :param cm_dict_: Dict of confusion matrices containing information
                        about reference and protected groups.
        :param control_level_: Name of reference group in cm_dict as a string.
        :param protected_level_: Name of protected group in cm_dict as a string.
        :return: AIR value.
    """

    # control group summary
    control_accepted = float(cm_dict_[control_level_].iat[0, 0] + cm_dict_[control_level_].iat[0, 1])
    control_total = float(cm_dict_[control_level_].sum().sum())
    control_prop = control_accepted / control_total

    # protected group summary
    protected_accepted = float(cm_dict_[protected_level_].iat[0, 0] + cm_dict_[protected_level_].iat[0, 1])
    protected_total = float(cm_dict_[protected_level_].sum().sum())
    protected_prop = protected_accepted / protected_total

    logger.info(str(control_level_).title() + ' proportion accepted: %.3f' % control_prop)
    logger.info(str(protected_level_).title() + ' proportion accepted: %.3f' % protected_prop)

    # return adverse impact ratio
    return float(protected_prop / control_prop)


def main():

    # set random seed
    seed(SEED)

    ####################################################################################################################
    # logging config

    # log file
    logger.setLevel(logging.DEBUG)

    # create console handler and file handler and set level to debug
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s.%(msecs)06d: %(levelname)s %(name)s:%(lineno)d:\n%(message)s',
                                  datefmt='%Y-%m-%d %I:%M:%S')

    # add formatter
    sh.setFormatter(formatter)

    # add handler to logger
    logger.addHandler(sh)

    ####################################################################################################################
    # output directory

    # output directory
    out_dir = 'out-' + PREFIX + '-' + time_stamp

    try:
        if not os.path.exists(out_dir):

            os.mkdir(out_dir)

            log_name = out_dir + os.sep + PREFIX + '-' + time_stamp + '.log'
            fh = logging.FileHandler(log_name)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            logger.info('Created output directory: %s' % out_dir)

    except IOError:
        print('Failed to create output directory: %s!' % out_dir)
        sys.exit(-1)

    ####################################################################################################################
    # training data

    iterations = pd.DataFrame(columns=['Main AUC', 'Main AIR', 'Adversary AUC'])

    train_f_name = '/home/patrickh/Workspace/GWU_rml/tests/data/train_simulated_transformed.csv'
    valid_f_name = '/home/patrickh/Workspace/GWU_rml/tests/data/test_simulated_transformed.csv'
    x_names = ['binary1', 'binary2', 'cat1_0', 'cat1_1', 'cat1_2', 'cat1_3', 'cat1_4', 'fried1_std', 'fried2_std',
               'fried3_std', 'fried4_std', 'fried5_std']
    y_name = 'outcome'

    demo_name = 'ctrl_class1'
    protected_level = 0
    control_level = 1


    ####################################################################################################################
    # training

    iter_frame = pd.DataFrame(columns=['Adv. AUC', 'Main AUC', 'Main AIR'])

    logger.info('Training ...')

    # initialize main model

    htrain = h2o.load_dataset(train_f_name)
    htrain[y_name] = htrain[y_name].asfactor()
    print(htrain.head())
    hvalid = h2o.load_dataset(valid_f_name)
    hvalid[y_name] = hvalid[y_name].asfactor()
    print(hvalid.head())

    main_ = model.gbm_grid(x_names, y_name, htrain, hvalid, SEED)
    main_auc = main_.auc(valid=True)

    acc = main_.accuracy(valid=True)
    logger.info('Initial GBM grid search completed with accuracy %.4f at threshold %.4f.' %
                (acc[0][1], acc[0][0]))

    hvalid['pred'] = main_.predict(hvalid)['p1']

    cm_protected = get_confusion_matrix(hvalid.as_data_frame(), y_name, 'pred', by=demo_name, level=protected_level,
                                        cutoff=acc[0][0])
    logger.info(cm_protected)

    cm_control = get_confusion_matrix(hvalid.as_data_frame(), y_name, 'pred', by=demo_name, level=control_level,
                                     cutoff=acc[0][0])
    logger.info(cm_control)

    cm_dict = {0: cm_protected, 1: cm_control}
    main_air = get_air(cm_dict, control_level, protected_level)

    logger.info('Initial GBM AIR %.4f.' % main_air)

    iter_frame = iter_frame.append({'Adv. AUC': np.nan,
                                    'Main AUC': main_auc,
                                    'Main AIR': main_air}, ignore_index=True)

    for i in range(0, EPOCHS):

        # train adversary to create weights

        adv_htrain = main_.predict_contributions(htrain)
        adv_htrain['pred'] = main_.predict(htrain)['p1']  # probability of high priced
        adv_htrain['demo'] = htrain[demo_name]
        adv_htrain['demo'] = adv_htrain['demo'].asfactor()
        print(adv_htrain.head())

        adv_hvalid = main_.predict_contributions(hvalid)
        adv_hvalid['pred'] = main_.predict(hvalid)['p1']  # probability of high priced
        adv_hvalid['demo'] = hvalid[demo_name]
        adv_hvalid['demo'] = adv_hvalid['demo'].asfactor()
        print(adv_hvalid.head())

        adversary_ = model.gbm_grid(x_names + ['pred'], 'demo', adv_htrain, adv_hvalid, SEED)
        adv_auc = adversary_.auc(valid=True)

        logger.info('Epoch %d adversary AUC: %.4f.' % (int(i), adv_auc))

        # re-train main model with weights

        htrain['weight'] = adversary_.predict(adv_htrain)['p0']  # probability of control
        print(htrain.head())
        hvalid['weight'] = adversary_.predict(adv_hvalid)['p0']  # probability of control
        print(hvalid.head())

        main_ = model.gbm_grid(x_names, y_name, htrain, hvalid, SEED, weight='weight')
        main_auc = main_.auc(valid=True)

        acc = main_.accuracy(valid=True)
        logger.info('Epoch %d GBM grid search completed with accuracy %.4f at threshold %.4f.' %
                    (int(i), acc[0][1], acc[0][0]))

        hvalid['pred'] = main_.predict(hvalid)['p1']

        cm_protected = get_confusion_matrix(hvalid.as_data_frame(), y_name, 'pred', by=demo_name, level=protected_level,
                                            cutoff=acc[0][0])
        logger.info(cm_protected)

        cm_control = get_confusion_matrix(hvalid.as_data_frame(), y_name, 'pred', by=demo_name, level=control_level,
                                          cutoff=acc[0][0])
        logger.info(cm_control)

        cm_dict = {0: cm_protected, 1: cm_control}
        main_air = get_air(cm_dict, control_level, protected_level)

        logger.info('Epoch %d GBM AIR %.4f.' % (int(i), main_air))

        iter_frame = iter_frame.append({'Adv. AUC': adv_auc,
                                        'Main AUC': main_auc,
                                        'Main AIR': main_air}, ignore_index=True)

        logger.info(iter_frame)

    iter_frame.to_csv(out_dir + os.sep + 'iter.csv')

    ####################################################################################################################
    # end timer

    toc = time.time() - tic
    logger.info('All tasks completed in %.2f s' % toc)


if __name__ == '__main__':
    main()

########################################################################################################################
# notes
