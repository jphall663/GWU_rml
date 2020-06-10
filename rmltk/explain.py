import h2o
from h2o.backend import H2OLocalServer
from h2o.estimators.random_forest import H2ORandomForestEstimator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess


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


def plot_coefs(coef_list, model_list, title_model, column_order):

    """ Plots global var. imp. importance coefficients stored in Pandas
        frames in coef_list.

    :param coef_list: List containing global var. imp. coefficients
                      for models in model list (tightly coupled to frame schemas).
    :param model_list: List of H2O GBM models trained in forward selection.
    :param title_model: Display name of model in coefficient plot.
    :param column_order: List of column names to preserve coloring
                         from previous coefficient plots.
    """

    for j, frame in enumerate(coef_list):

        auc_ = model_list[j].auc(valid=True)
        title_ = title_model + ' Model: {j}\n GBM AUC: {auc:.2f}'.format(j=str(j + 1), auc=auc_)
        fig, ax_ = plt.subplots(figsize=(10, 8))
        _ = frame[column_order].plot(kind='barh',
                                     ax=ax_,
                                     title=title_,
                                     colormap='gnuplot')


def pd_ice(x_name, valid, model, resolution=20, bins=None):

    """ Creates Pandas DataFrame containing partial dependence or ICE
        for a single input variable.

    :param x_name: Variable for which to calculate partial dependence.
    :param valid: Pandas validation frame.
    :param model: H2O model (assumes binary classifier).
    :param resolution: The number of points across the domain of xs for which
                       to calculate partial dependence, default 20.
    :param bins: List of values at which to set xs, default 20 equally-spaced
                 points between column minimum and maximum.

    :return: Pandas DataFrame containing partial dependence values.

    """

    # turn off pesky Pandas copy warning
    pd.options.mode.chained_assignment = None

    # determine values at which to calculate partial dependence
    if bins is None:
        min_ = valid[x_name].min()
        max_ = valid[x_name].max()
        by = (max_ - min_) / resolution
        # modify max and by
        # to preserve resolution and actually search up to max
        bins = np.arange(min_, (max_ + by), (by + np.round((1. / resolution) * by, 3)))

        # cache original column values
    col_cache = valid.loc[:, x_name].copy(deep=True)

    # calculate partial dependence
    # by setting column of interest to constant
    # and scoring the altered data and taking the mean of the predictions
    temp_df = valid.copy(deep=True)
    temp_df.loc[:, x_name] = bins[0]
    for j, _ in enumerate(bins):
        if j + 1 < len(bins):
            valid.loc[:, x_name] = bins[j + 1]
            temp_df = temp_df.append(valid, ignore_index=True)

    # return input frame to original cached state
    valid.loc[:, x_name] = col_cache

    # model predictions
    # probably assumes binary classification
    temp_df['partial_dependence'] = model.predict(h2o.H2OFrame(temp_df))['p1'].as_data_frame()

    return pd.DataFrame(temp_df[[x_name, 'partial_dependence']].groupby([x_name]).mean()).reset_index()


def get_percentile_dict(yhat_name, valid, id_):

    """ Returns the percentiles of a column, yhat_name, as the indices based on
        another column id_.

    :param yhat_name: Name of column in valid in which to find percentiles.
    :param valid: Pandas validation frame.
    :param id_: Validation Pandas frame containing yhat and id_.

    :return: Dictionary of percentile values and index column values.

    """

    # create a copy of frame and sort it by yhat
    sort_df = valid.copy(deep=True)
    sort_df.sort_values(yhat_name, inplace=True)
    sort_df.reset_index(inplace=True)

    # find top and bottom percentiles
    percentiles_dict = {0: sort_df.loc[0, id_], 99: sort_df.loc[sort_df.shape[0] - 1, id_]}

    # find 10th-90th percentiles
    inc = sort_df.shape[0] // 10
    for i in range(1, 10):
        percentiles_dict[i * 10] = sort_df.loc[i * inc, id_]

    return percentiles_dict


def plot_pd_ice(x_name, par_dep_frame, ax=None):

    """ Plots ICE overlayed onto partial dependence for a single variable.
    Conditionally uses user-defined axes, ticks, and labels for grouped subplots.

    :param x_name: Name of variable for which to plot ICE and partial dependence.
    :param par_dep_frame: Name of Pandas frame containing ICE and partial
                          dependence values (tightly coupled to frame schema).
    :param ax: Matplotlib axis object to use.
    """

    # for standalone plotting
    if ax is None:

        # initialize figure and axis
        fig, ax = plt.subplots()

        # plot ICE
        par_dep_frame.drop('partial_dependence', axis=1).plot(x=x_name,
                                                              colormap='gnuplot',
                                                              ax=ax)
        # overlay partial dependence, annotate plot
        par_dep_frame.plot(title='Partial Dependence with ICE: ' + x_name,
                           x=x_name,
                           y='partial_dependence',
                           color='grey',
                           linewidth=3,
                           ax=ax)

    # for grouped subplots
    else:

        # plot ICE
        par_dep_frame.drop('partial_dependence', axis=1).plot(x=x_name,
                                                              colormap='gnuplot',
                                                              ax=ax)

        # overlay partial dependence, annotate plot
        par_dep_frame.plot(title='Partial Dependence with ICE: ' + x_name,
                           x=x_name,
                           y='partial_dependence',
                           color='red',
                           linewidth=3,
                           ax=ax)


def hist_mean_pd_ice_plot(x_name, y_name, valid, pd_ice_dict):

    """ Plots diagnostic plot of histogram with mean line overlay
        side-by-side with partial dependence and ICE.

    :param x_name: Name of variable for which to plot ICE and partial dependence.
    :param y_name: Name of target variable.
    :param valid: Pandas validation frame.
    :param pd_ice_dict: Dict of Pandas DataFrames containing partial dependence
                        and ICE values.
    """

    # initialize figure and axis
    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=False)
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1.8, wspace=0.18)

    # if variable is *not* high cardinality
    # create histogram directly
    if valid[x_name].nunique() <= 20:
        mean_df = valid[[x_name, y_name]].groupby(by=x_name).mean()
        freq, bins, _ = ax.hist(valid[x_name], color='k')

    # if variable is high cardinality
    # bin, then create hist
    else:
        temp_df = pd.concat([pd.cut(valid[x_name], pd_ice_dict[x_name][x_name] - 1, duplicates='drop'),
                             valid[y_name]], axis=1)
        mean_df = temp_df.groupby(by=x_name).mean()
        del temp_df
        freq, bins, _ = ax.hist(valid[x_name], bins=pd_ice_dict[x_name][x_name] - 1, color='k')
        bins = bins[:-1]

    # annotate hist
    ax.set_xlabel(x_name)
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram with Mean ' + y_name + ' Overlay')

    # create a new twin axis
    # on which to plot a line showing mean value
    # across hist bins
    ax1 = ax.twinx()
    _ = ax1.set_ylim((0, 1))
    _ = ax1.plot(bins, mean_df.reindex(labels=bins)[y_name], color='r')
    _ = ax1.set_ylabel('Mean ' + y_name)
    _ = ax1.legend(['Mean ' + y_name], loc=1)

    # plot PD and ICE
    plot_pd_ice(x_name,
                pd_ice_dict[x_name],
                ax2)
    _ = ax2.legend(bbox_to_anchor=(1.05, 0),
                   loc=3,
                   borderaxespad=0.)


def get_gv(title, model_id, mojo_path):
    """ Utility function to generate graphviz dot file from h2o MOJO using
        a subprocess.

    Args:
        title: Title for displayed decision tree.
        model_id: h2o model identifier.
        mojo_path: Path to saved model MOJO (Java scoring artifact);
                   generated by train_cv_dt function above.

    """

    # locate h2o jar
    hs = H2OLocalServer()
    h2o_jar_path = hs._find_jar()
    print('Discovered H2O jar path:\n', h2o_jar_path)

    # construct command line call to generate graphviz version of
    # tree, see for more information:
    # http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html
    gv_file_name = model_id + '.gv'
    gv_args = str('-cp ' + h2o_jar_path +
                  ' hex.genmodel.tools.PrintMojo --tree 0 -i '
                  + mojo_path + ' -o').split()
    gv_args.insert(0, 'java')
    gv_args.append(gv_file_name)
    if title is not None:
        gv_args = gv_args + ['--title', title]

    # call constructed command
    print()
    print('Calling external process ...')
    print(' '.join(gv_args))
    # if the line below is failing for you, try instead:
    # _ = subprocess.call(gv_args, shell=True)
    _ = subprocess.call(gv_args)


def get_png(model_id):

    """ Utility function to generate PNGs from .dots using a subprocess.

    Arg:
        model_id: h2o model identifier.

    """

    gv_file_name = model_id + '.gv'

    # construct call to generate PNG from
    # graphviz representation of the tree
    png_file_name = model_id + '.png'
    png_args = str('dot -Tpng ' + gv_file_name + ' -o ' + png_file_name)
    png_args = png_args.split()

    # call
    print('Calling external process ...')
    print(' '.join(png_args))
    # if the line below is failing for you, try instead:
    # _ = subprocess.call(png_args, shell=True)
    _ = subprocess.call(png_args)


def get_cv_dt(x_names, y_names, train, model_id, seed_, title, valid=None):

    """ Utility function to train decision trees.

    Args:
        model_id: h2o model identifier.
        frame: Pandas DataFrame containing X and yhat on which to train
               the decision trees.

    Returns:
        Model as h2o model object.

    """

    # initialize single tree model
    tree = H2ORandomForestEstimator(ntrees=1,  # use only one tree
                                    sample_rate=1,  # use all rows in that tree
                                    mtries=-2,  # use all columns in that tree's split search
                                    max_depth=4,  # shallow trees are easier to understand
                                    seed=seed_,  # set random seed for reproducibility
                                    nfolds=3,  # cross-validation for stability and ...
                                    # only way to get metrics for 1 tree in h2o
                                    model_id=model_id)  # gives MOJO artifact a recognizable name

    # train single tree model
    if valid is not None:
        tree.train(x=x_names, y=y_names, training_frame=h2o.H2OFrame(train), validation_frame=h2o.H2OFrame(valid))
    else:
        tree.train(x=x_names, y=y_names, training_frame=h2o.H2OFrame(train))

    # persist MOJO (compiled Java representation of trained model)
    # from which to generate plot of tree
    mojo_path = tree.download_mojo(path='.')
    print('Generated MOJO path:\n', mojo_path)

    get_gv(title, model_id, mojo_path)
    get_png(model_id)

    return tree
