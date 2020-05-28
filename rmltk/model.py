import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
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
# TODO: XNN
# TODO: NMF model
# TODO: model documentation


def glm_grid(x_names, y_name, htrain, hvalid, seed_, weight=None):

    """ Wrapper function for penalized GLM with alpha and lambda search.

    :param x_names: List of inputs.
    :param y_name: Name of target variable.
    :param htrain: Training H2OFrame.
    :param hvalid: Validation H2OFrame.
    :param seed_: Random seed for better reproducibility.
    :param weight:
    :return: Best H2OGeneralizedLinearEstimator.
    """

    alpha_opts = [0.01, 0.25, 0.5, 0.99]  # always keep some L2

    # define search criteria
    # i.e., over alpha
    # lamda search handled by lambda_search param below
    hyper_parameters = {'alpha': alpha_opts}

    # initialize grid search
    grid = H2OGridSearch(
        H2OGeneralizedLinearEstimator(family="binomial",
                                      lambda_search=True,
                                      seed=seed_),
        hyper_params=hyper_parameters)

    # execute training w/ grid search
    grid.train(y=y_name,
               x=x_names,
               training_frame=htrain,
               validation_frame=hvalid,
               weights_column=weight,
               seed=seed_)

    # select best model from grid search
    best_model = grid.get_grid()[0]
    del grid

    return best_model


def gbm_grid(x_names, y_name, htrain, hvalid, seed_, weight=None,
             monotone_constraints_=None, hyper_params_=None,
             search_criteria_=None):

    """ Wrapper that trains a random grid of H2OGradientBoostingEstimators,
        optionally with user-designated monotonicity constraints, hyper_params,
        and search criteria.

    :param x_names: List of inputs.
    :param y_name: Name of target variable.
    :param htrain: Training H2OFrame.
    :param hvalid: Validation H2OFrame.
    :param seed_: Random seed for better reproducibility.
    :parem weight:
    :param monotone_constraints_: Dictionary of monotonicity constraints (optional).
    :param hyper_params_: Dictionary of hyperparamters over which to search (optional).
    :param search_criteria_: Dictionary of criterion for grid search (optional).
    :return: Best H2OGeneralizedLinearEstimator.
    """

    # define default random grid search parameters
    if hyper_params_ is None:

        hyper_params_ = {'ntrees': list(range(1, 500, 50)),
                         'max_depth': list(range(1, 20, 2)),
                         'sample_rate': [s / float(10) for s in range(1, 11)],
                         'col_sample_rate': [s / float(10) for s in range(1, 11)]}

    # define default search strategy
    if search_criteria_ is None:

        search_criteria_ = {'strategy': 'RandomDiscrete',
                            'max_models': 20,
                            'max_runtime_secs': 600,
                            'seed': seed_}

    # initialize grid search
    grid = H2OGridSearch(H2OGradientBoostingEstimator,
                         hyper_params=hyper_params_,
                         search_criteria=search_criteria_)

    # execute training w/ grid search
    grid.train(x=x_names,
               y=y_name,
               monotone_constraints=monotone_constraints_,
               training_frame=htrain,
               validation_frame=hvalid,
               stopping_rounds=5,
               weights_column=weight,
               seed=seed_)

    # select best model from grid search
    best_model = grid.get_grid()[0]
    del grid

    return best_model


def gbm_forward_select_train(orig_x_names, y_name, train, valid, seed_, next_list,
                             coef_frame, new_col_name, monotone=False, monotone_constraints_=None,
                             hyper_params_=None, search_criteria_=None):

    """Trains multiple GBMs based on forward selection, optionally with user-designated
       monotonicity constraints, hyper_params, and search criteria.

    :param orig_x_names: List of inputs to include in first model and
                         from which to start forward selection process.
    :param y_name: Name of target variable.
    :param train: Pandas training frame.
    :param valid: Pandas validation frame.
    :param seed_: Random seed for better reproducibility.
    :param next_list: List of features for forward selection process.
    :param coef_frame: Pandas frame of previous model global var. imp.
                       coefficients (tightly coupled to frame schema).
    :param new_col_name: Name in coef_frame for column for this training
                         run's global var. imp. coefficients.
    :param monotone: Whether or not to create monotonic GBMs.
    :param monotone_constraints_: Dictionary of monotonicity constraints (optional).
    :param hyper_params_: Dictionary of hyperparamters over which to search (optional).
    :param search_criteria_: Dictionary of criterion for grid search (optional).
    :return: Dictionary of: list of H2O GBM models trained in forward selection, list
             containing a coef_frame for each model, list of Shapley values for each model.
    """

    # init empty parallel lists to store results
    model_list = []
    coef_list = []
    shap_list = []

    # init loop var
    selected = orig_x_names

    for j in range(0, len(next_list) + 1):

        # init or clear local dict of monotone constraints
        mc = None

        # optionally select or generate mc
        if monotone:

            if monotone_constraints_ is None:
                # create mc anew for the current model using Pearson correlation
                names = list(valid[selected + [y_name]].corr()[y_name].index)[:-1]
                signs = list([int(i) for i in np.sign(valid[selected + [y_name]].corr()[y_name].values[:-1])])
                mc = dict(zip(names, signs))
            else:
                # select mc from user designated dict: monotone_constraints_
                mc = {name_: monotone_constraints_[name_] for name_ in selected}

        # convert training and test data to h2o format
        # necessary to ensure ordering of Shapley values matches selected
        # ensure y is treated as binomial
        htrain = h2o.H2OFrame(train[selected + [y_name]])
        htrain[y_name] = htrain[y_name].asfactor()
        hvalid = h2o.H2OFrame(valid[selected + [y_name]])

        # train model and calculate Shapley values
        print('Starting grid search %i/%i ...' % (j + 1, len(next_list)+1))
        print('Input features =', selected)
        if mc is not None:
            print('Monotone constraints =', mc)
        model_list.append(gbm_grid(selected, y_name, htrain, hvalid, seed_,
                                   monotone_constraints_=mc, hyper_params_=hyper_params_,
                                   search_criteria_=search_criteria_))
        shap_values = model_list[j].predict_contributions(hvalid).as_data_frame().values[:, :-1]
        shap_list.append(shap_values)

        # update coef_frame with current model Shapley values
        # update coef_list
        col = pd.DataFrame({new_col_name: list(np.abs(shap_values).mean(axis=0))}, index=selected)
        coef_frame.update(col)
        coef_list.append(coef_frame.copy(deep=True))  # deep copy necessary

        # retrieve AUC and update progress
        auc_ = model_list[j].auc(valid=True)
        print('Completed grid search %i/%i with AUC: %.2f ...' % (j + 1, len(next_list)+1, auc_))
        print('--------------------------------------------------------------------------------')

        # add the next most y-correlated feature
        # for the next modeling iteration
        if j < len(next_list):
            selected = selected + [next_list[j]]

    print('Done.')

    return {'MODELS': model_list, 'GLOBAL_COEFS': coef_list, 'LOCAL_COEFS': shap_list}
