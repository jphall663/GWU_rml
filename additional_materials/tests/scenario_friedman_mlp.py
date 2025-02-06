# Trains a simple NN on the Friedman data

# Python imports
import json
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import logging
from matplotlib import pyplot
import matplotlib.patches as mpatches
from numpy.random import seed
import numpy as np
import pandas as pd
import random
from tensorflow import set_random_seed
from tensorflow.keras import regularizers
import os
import sys
import time

# TODO: incorporate into rmltk
# TODO: main routines into functions/ann super class/mlp/xnn subclasses
# TODO: unit tests
# TODO: command line args
# TODO: custom loss with AIR
# TODO: XNN
# TODO: PD/ICE
# TODO: local explanations

# global training constants
EPOCHS = 50
PATIENCE = 5
SEED = 33333
N_MODELS = 3

"""

BATCH_SIZE = 128
N_LAYER = 2
N_UNIT = 12
ACTIVATION_TYPE = 'relu'
DROPOUT = 0.2
L2 = 1e-3

"""

BATCH_SIZE = [32, 64, 128, 256, 512]
N_LAYER = [1, 2, 3]
N_UNIT = [6, 12, 18, 24]
ACTIVATION_TYPE = ['relu', 'tanh']
DROPOUT = [0.0, 0.01, 0.1, 0.2, 0.3]
L2 = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# global system constants
PREFIX = 'friedman-mlp'
VERBOSE = False
FOLD_NAME = None

# start timer
tic = time.time()

# time stamp
time_stamp = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(tic))

# init logger
logger = logging.getLogger(__file__)


def load_data(f_name, x_names_, y_name_, fold_name_, pandas_where_=None, logger_=logger):

    frame = pd.read_csv(f_name)

    if pandas_where_ is not None:

        query = 'frame[' + pandas_where_ + ']'
        logger_.info('Evaluating Pandas query: %s' % query)
        frame = eval(query)

    n_instances_, n_features_ = frame[x_names_].shape
    n_y_levels_ = int(frame[y_name_].nunique())

    folds_ = 0
    if fold_name_ is not None:
        folds_ = sorted(list(frame[fold_name_].unique()))

    logger.info('Loaded file %s with:\n%i features\n%i instances\n%i target levels\n%s folds' % (f_name,
                                                                                                 int(n_features_),
                                                                                                 int(n_instances_),
                                                                                                 int(n_y_levels_),
                                                                                                 str(folds_)))

    return {'X': frame[x_names_].values,
            'p': n_features_,
            'N': n_instances_,
            'y': keras.utils.to_categorical(frame[y_name_].values, n_y_levels_),
            'N_y_levels': n_y_levels_,
            'folds': folds_}


def init_arch(n_y_levels_, n_features_, units=None, activations=None, dropout=None, l2=None):

    model_ = Sequential()

    for i in range(0, len(units)):

        if i == 0:
            model_.add(Dense(units[i], activation=activations[i], input_shape=(n_features_,),
                             kernel_regularizer=regularizers.l2(l2[i])))
            model_.add(Dropout(dropout[i]))
        else:
            model_.add(Dense(units[i], activation=activations[i], kernel_regularizer=regularizers.l2(l2[i])))
            model_.add(Dropout(dropout[i]))

    model_.add(Dense(n_y_levels_, activation='softmax'))

    model_.compile(loss='categorical_crossentropy',
                   optimizer=RMSprop(),
                   metrics=['accuracy'])

    return model_


def get_prauc(frame_, y_name_, yhat_name_, pos=1, neg=0, res=0.01):

    """ Calculates precision, recall, and f1 for a pandas dataframe of y and yhat values.

    Args:
        frame_: Pandas dataframe of actual (y) and predicted (yhat) values.
        y_name_: Name of actual value column.
        yhat_name_: Name of predicted value column.
        pos: Primary target value, default 1.
        neg: Secondary target value, default 0.
        res: Resolution by which to loop through cutoffs, default 0.01.

    Returns:
        Pandas dataframe of precision, recall, and f1 values.
    """

    temp_df = frame_.copy(deep=True)  # don't destroy original data
    dname = 'd_' + str(y_name_)  # column for predicted decisions
    eps = 1e-20  # for safe numerical operations

    # init p-r roc frame
    prauc_frame_ = pd.DataFrame(columns=['cutoff', 'recall', 'precision', 'f1'])

    # loop through cutoffs to create p-r roc frame
    for cutoff in np.arange(0, 1 + res, res):

        # binarize decision to create confusion matrix values
        temp_df[dname] = np.where(temp_df[yhat_name_] > cutoff, 1, 0)

        # calculate confusion matrix values
        tp = temp_df[(temp_df[dname] == pos) & (temp_df[y_name_] == pos)].shape[0]
        fp = temp_df[(temp_df[dname] == pos) & (temp_df[y_name_] == neg)].shape[0]
        fn = temp_df[(temp_df[dname] == neg) & (temp_df[y_name_] == pos)].shape[0]

        # calculate precision, recall, and f1
        recall = (tp + eps) / ((tp + fn) + eps)
        precision = (tp + eps) / ((tp + fp) + eps)
        f1 = 2 / ((1 / (recall + eps)) + (1 / (precision + eps)))

        # add new values to frame
        prauc_frame_ = prauc_frame_.append({'cutoff': cutoff,
                                            'recall': recall,
                                            'precision': precision,
                                            'f1': f1},
                                           ignore_index=True)

    # housekeeping
    del temp_df

    return prauc_frame_


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
    set_random_seed(SEED)

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

    # log file


    ####################################################################################################################
    # load training data

    logger.info('Loading data ...')

    train_f_name = 'data/train_simulated_transformed.csv'
    test_f_name = 'data/test_simulated_transformed.csv'
    x_names = ['binary1', 'binary2', 'cat1_0', 'cat1_1', 'cat1_2', 'cat1_3', 'cat1_4', 'fried1_std', 'fried2_std',
               'fried3_std', 'fried4_std', 'fried5_std']
    y_name = 'outcome'

    demo_name = 'ctrl_class1'
    protected_level = 0
    control_level = 1

    train_data_dict = load_data(train_f_name, x_names, y_name, FOLD_NAME)
    x_train = train_data_dict['X']
    n_features = train_data_dict['p']
    y_train = train_data_dict['y']
    n_y_levels = train_data_dict['N_y_levels']
    folds = train_data_dict['folds']

    test_data_dict = load_data(test_f_name, x_names, y_name, None)
    x_test = test_data_dict['X']
    y_test = test_data_dict['y']

    ####################################################################################################################
    # training

    logger.info('Training ...')

    # set callback for early stopping and check-pointing
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=int(VERBOSE), patience=PATIENCE)

    ####################################################################################################################
    # random grid search

    # init grid search storage
    grid_dict = {}
    grid_df = pd.DataFrame(columns=['run', 'acc', 'air', 'batch_size', 'layers', 'units', 'activations', 'dropout',
                                    'l2'])

    for run in range(0, N_MODELS):

        # weight file name must reflect grid run and not be overwritten
        weights_out_f_name = PREFIX + '-' + time_stamp + '-weights.h5'

        # init run storage
        run_dict = {}
        grid_dict[run] = run_dict

        if N_MODELS == 1:

            # default tunable params
            batch_size = BATCH_SIZE
            model_struct = {'units': [N_UNIT] * N_LAYER,
                            'activations': [ACTIVATION_TYPE] * N_LAYER,
                            'dropout': [DROPOUT] * N_LAYER,
                            'l2': [L2] * N_LAYER}

        else:

            logger.info('Grid search run: %i of %i.' % (int(run + 1), int(N_MODELS)))

            # weight file name must reflect grid run and not be overwritten
            weights_out_f_name = PREFIX + '-' + time_stamp + '-weights.h5'
            weights_out_f_name = 'grid-' + str(run) + '-' + weights_out_f_name

            batch_size = random.choice(BATCH_SIZE)
            layers = random.choice(N_LAYER)
            model_struct = {'units': random.choices(N_UNIT, k=layers),
                            'activations': [random.choice(ACTIVATION_TYPE)] * layers,
                            'dropout': [random.choice(DROPOUT)] * layers,
                            'l2': [random.choice(L2)] * layers}

        # with cross-validation
        if FOLD_NAME is not None:

            fold_dict = {}

            for fold in folds:

                # init model
                model = init_arch(n_y_levels, n_features, **model_struct)

                # view model architecture
                model.summary()

                logger.info('Cross-validation fold: %i of %i' % (int(fold), len(folds)))

                cv_weights_out_f_name = 'cv-' + str(fold) + '-' + weights_out_f_name
                fold_dict[fold] = {}
                fold_dict[fold]['weights'] = cv_weights_out_f_name

                pandas_where = 'frame["' + str(FOLD_NAME) + '"] != ' + str(fold)
                cv_train_data_dict = load_data(train_f_name, x_names, y_name, None, pandas_where_=pandas_where)
                cv_x_train = cv_train_data_dict['X']
                cv_y_train = cv_train_data_dict['y']

                pandas_where = 'frame["' + str(FOLD_NAME) + '"] == ' + str(fold)
                cv_valid_data_dict = load_data(train_f_name, x_names, y_name, None, pandas_where_=pandas_where)
                cv_x_valid = cv_valid_data_dict['X']
                cv_y_valid = cv_valid_data_dict['y']

                # set callback for model check point
                cp = ModelCheckpoint(out_dir + os.sep + cv_weights_out_f_name, verbose=int(VERBOSE), monitor='val_loss',
                                     save_best_only=True, mode='auto')

                # train
                history = model.fit(cv_x_train, cv_y_train,
                                    batch_size=batch_size,
                                    epochs=EPOCHS,
                                    verbose=int(VERBOSE),
                                    validation_data=(cv_x_valid, cv_y_valid),
                                    callbacks=[es, cp])

                fold_dict[fold]['history'] = history

                # reset for next folds
                del model
                del history

        # without cross-validation
        else:

            # init model
            model = init_arch(n_y_levels, n_features, **model_struct)

            # view model architecture
            model.summary()

            # set callback for model check point
            cp = ModelCheckpoint(out_dir + os.sep + weights_out_f_name, verbose=int(VERBOSE), monitor='val_loss',
                                 save_best_only=True, mode='auto')
            # train
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=EPOCHS,
                                verbose=int(VERBOSE),
                                validation_data=(x_test, y_test),
                                callbacks=[es, cp])

            del model

        ################################################################################################################
        # iteration chart

        logger.info('Plotting iteration history ...')

        if FOLD_NAME is not None:

            for fold in folds:
                pyplot.plot(fold_dict[fold]['history'].history['loss'], color='b')
                label1 = 'Fold ' + str(fold) + ' Train Loss'

                pyplot.plot(fold_dict[fold]['history'].history['val_loss'], color='g')
                label2 = 'Fold ' + str(fold) + ' Validation Loss'

                b_patch = mpatches.Patch(color='b', label=label1)
                g_patch = mpatches.Patch(color='g', label=label2)

                pyplot.legend(handles=[b_patch, g_patch])

        else:

            pyplot.plot(history.history['loss'], color='b')
            label1 = 'Training Loss'

            pyplot.plot(history.history['val_loss'], color='g')
            label2 = 'Validation Loss'

            pyplot.vlines(np.asarray(history.history['val_loss']).argmin(), es.best,
                          history.history['loss'][0], color='k')
            label3 = 'Best Epoch'

            b_patch = mpatches.Patch(color='b', label=label1)
            g_patch = mpatches.Patch(color='g', label=label2)
            k_patch = mpatches.Patch(color='k', label=label3)

            pyplot.legend(handles=[b_patch, g_patch, k_patch])

            # clean up for next model in grid search
            del history

        pyplot.title('Iteration Plot')

        if N_MODELS > 1:
            plot_out_f_name = out_dir + os.sep + 'grid-' + str(run) + '-' + PREFIX + '-' + time_stamp + '.png'
            pyplot.savefig(plot_out_f_name)
            logger.info('Plot written to: %s' % plot_out_f_name)
        else:
            plot_out_f_name = out_dir + os.sep + PREFIX + '-' + time_stamp + '.png'
            pyplot.savefig(plot_out_f_name)
            logger.info('Plot written to: %s' % plot_out_f_name)

        pyplot.clf()

        ################################################################################################################
        # cutoff

        logger.info('Selecting cutoff ...')

        # load test data (again!?)
        test_yhat = pd.read_csv(test_f_name)[x_names + [demo_name, y_name]]
        yhat_name = 'p_' + y_name

        # create empty model to hold best weights
        model_ckpt: Sequential = init_arch(n_y_levels, n_features, **model_struct)

        if FOLD_NAME is not None:

            # retrieve best model weights from checkpoints
            cv_yhat_names = []
            for fold in folds:
                model_ckpt.load_weights(out_dir + os.sep + fold_dict[fold]['weights'])
                cv_yhat_name = 'cv_' + str(fold) + '_' + yhat_name
                cv_yhat_names.append(cv_yhat_name)
                test_yhat[cv_yhat_name] = model_ckpt.predict(test_yhat[x_names].values, verbose=int(VERBOSE))[:, 1]

            test_yhat[yhat_name] = test_yhat[cv_yhat_names].mean(axis=1)

        else:

            # retrieve best model weights from checkpoint
            model_ckpt.load_weights(out_dir + os.sep + weights_out_f_name)
            test_yhat[yhat_name] = model_ckpt.predict(test_yhat[x_names].values)[:, 1]

        prauc_frame = get_prauc(test_yhat, y_name, yhat_name)
        best_cut = prauc_frame.loc[prauc_frame['f1'].idxmax(), 'cutoff']  # Find cutoff w/ max F1

        logger.info('Maximum F1 threshold: %.8f' % best_cut)

        ################################################################################################################
        # confusion matrices

        logger.info('Generating confusion matrices ...')

        cm = get_confusion_matrix(test_yhat, y_name, yhat_name, cutoff=best_cut)
        acc = (cm.iat[0, 0] + cm.iat[1, 1]) / x_test.shape[0]

        logger.info(cm)
        logger.info('Accuracy at threshold = %.8f: %.4f' % (best_cut, acc))

        ################################################################################################################
        # AIR

        logger.info('Calculating AIR ...')

        cm_protected = get_confusion_matrix(test_yhat, y_name, yhat_name, by=demo_name, level=protected_level,
                                            cutoff=best_cut)
        logger.info(cm_protected)

        cm_control = get_confusion_matrix(test_yhat, y_name, yhat_name, by=demo_name, level=control_level,
                                          cutoff=best_cut)
        logger.info(cm_control)

        cm_dict = {0: cm_protected, 1: cm_control}
        air = get_air(cm_dict, control_level, protected_level)

        logger.info('AIR for %s, %s vs. %s: %.2f' % (demo_name, str(protected_level), str(control_level), air))

        ################################################################################################################
        # store run info

        if N_MODELS > 1:

            grid_dict[run]['model_struct'] = model_struct
            grid_dict[run]['batch_size'] = batch_size
            if FOLD_NAME is not None:
                grid_dict[run]['weights'] = {}
                for fold in folds:
                    grid_dict[run]['weights'][fold] = fold_dict[fold]['weights']
            else:
                grid_dict[run]['weights'] = weights_out_f_name
            grid_dict[run]['acc'] = acc
            grid_dict[run]['air'] = air

            grid_df = grid_df.append({'run': run,
                                      'acc': acc,
                                      'air': air,
                                      'batch_size': batch_size,
                                      'layers': len(model_struct['units']),
                                      'units': model_struct['units'],
                                      'activations': model_struct['activations'][0],
                                      'dropout': model_struct['dropout'][0],
                                      'l2': model_struct['l2'][0]},
                                     ignore_index=True)

            # serialize grid info
            with open(out_dir + os.sep + PREFIX + '-' + time_stamp + '.json', 'w') as f:
                json.dump(grid_dict, f)
            logger.info('Model information written to: %s' % out_dir + os.sep + PREFIX + '-' + time_stamp + '.json')

            grid_df.to_csv(out_dir + os.sep + PREFIX + '-' + time_stamp + '.csv')
            logger.info('Model information written to: %s' % out_dir + os.sep + PREFIX + '-' + time_stamp + '.csv')
            logger.info('Grid search models sorted by accuracy:')
            logger.info(grid_df.sort_values(by='acc', ascending=False))

            logger.info('Grid search models sorted by AIR:')
            logger.info(grid_df.sort_values(by='air', ascending=False))

    ####################################################################################################################
    # end timer

    toc = time.time() - tic
    logger.info('All tasks completed in %.2f s' % toc)


if __name__ == '__main__':
    main()

########################################################################################################################
# notes

# CV=5 EPOCHS=500 PATIENCE=50; Accuracy at threshold = 0.43000000: 0.7610; AIR for ctrl_class1, 0 vs. 1: 0.88
# CV=5 EPOCHS=500 PATIENCE=50 BATCH_SIZE=64; Accuracy at threshold = 0.43000000: 0.7633: 0.7610; AIR for ctrl_class1,
# 0 vs. 1: 0.89
# CV=5 EPOCHS=500 PATIENCE=50; Accuracy at threshold = 0.43000000: 0.7614; AIR for ctrl_class1, 0 vs. 1: 0.87
