import pandas as pd
import numpy as np
import string

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
# TODO: bring in model tester from ml-algos-perf
# TODO: additional discrimination metrics
# TODO: model documentation
# TODO: CV metrics for all measures, not just accuracy

# represent metrics as dictionary for use later
METRIC_DICT = {

    #### overall performance
    'Prevalence': '(tp + fn) / (tp + tn +fp + fn)',  # how much default actually happens for this group
    'Accuracy': '(tp + tn) / (tp + tn + fp + fn)',
    # how often the model predicts default and non-default correctly for this group

    #### predicting default will happen
    # (correctly)
    'True Positive Rate': 'tp / (tp + fn)',
    # out of the people in the group *that did* default, how many the model predicted *correctly* would default
    'Precision': 'tp / (tp + fp)',
    # out of the people in the group the model *predicted* would default, how many the model predicted *correctly* would default

    #### predicting default won't happen
    # (correctly)
    'Specificity': 'tn / (tn + fp)',
    # out of the people in the group *that did not* default, how many the model predicted *correctly* would not default
    'Negative Predicted Value': 'tn / (tn + fn)',
    # out of the people in the group the model *predicted* would not default, how many the model predicted *correctly* would not default

    #### analyzing errors - type I
    # false accusations
    'False Positive Rate': 'fp / (tn + fp)',
    # out of the people in the group *that did not* default, how many the model predicted *incorrectly* would default
    'False Discovery Rate': 'fp / (tp + fp)',
    # out of the people in the group the model *predicted* would default, how many the model predicted *incorrectly* would default

    #### analyzing errors - type II
    # costly ommisions
    'False Negative Rate': 'fn / (tp + fn)',
    # out of the people in the group *that did* default, how many the model predicted *incorrectly* would not default
    'False Omissions Rate': 'fn / (tn + fn)'
    # out of the people in the group the model *predicted* would not default, how many the model predicted *incorrectly* would not default
}


def get_metrics_ratios(cm_dict, _control_level):
    """ Calculates confusion matrix metrics in METRIC_DICT for each level of demographic feature.
    Tightly coupled to cm_dict.

    :param cm_dict: Dictionary of Pandas confusion matrices, one matrix for each level.
    :param _control_level: Control level in cm_dict.
    :return: Tuple, Pandas frame of metrics for each level of demographic feature, Pandas frame of ratio metrics for
             each level of demographic feature.

    """

    levels = sorted(list(cm_dict.keys()))

    eps = 1e-20  # for safe numerical operations

    # init return frames
    metrics_frame = pd.DataFrame(index=levels)  # frame for metrics

    # nested loop through:
    # - levels
    # - metrics
    for level in levels:

        for metric in METRIC_DICT.keys():
            # parse metric expressions into executable Pandas statements
            expression = METRIC_DICT[metric].replace('tp', 'cm_dict[level].iat[0, 0]') \
                .replace('fp', 'cm_dict[level].iat[0, 1]') \
                .replace('fn', 'cm_dict[level].iat[1, 0]') \
                .replace('tn', 'cm_dict[level].iat[1, 1]')

            # dynamically evaluate metrics to avoid code duplication
            metrics_frame.loc[level, metric] = eval(expression)

    # calculate metric ratios
    ratios_frame = (metrics_frame.loc[:, :] + eps) / (metrics_frame.loc[_control_level, :] + eps)
    ratios_frame.columns = [col + ' Ratio' for col in ratios_frame.columns]

    return metrics_frame, ratios_frame


def air(cm_dict, reference, protected):
    """ Calculates the adverse impact ratio as a quotient between protected and
        reference group acceptance rates: protected_prop/reference_prop.
        Prints intermediate values. Tightly coupled to cm_dict.

        :param cm_dict: Dict of confusion matrices containing information
                        about reference and protected groups.
        :param reference: Name of reference group in cm_dict as a string.
        :param protected: Name of protected group in cm_dict as a string.
        :return: AIR value.
    """

    # reference group summary
    reference_accepted = float(cm_dict[reference].iat[1, 0] + cm_dict[reference].iat[1, 1])  # predicted 0's
    reference_total = float(cm_dict[reference].sum().sum())
    reference_prop = reference_accepted / reference_total
    print(reference.title() + ' proportion accepted: %.3f' % reference_prop)

    # protected group summary
    protected_accepted = float(cm_dict[protected].iat[1, 0] + cm_dict[protected].iat[1, 1])  # predicted 0's
    protected_total = float(cm_dict[protected].sum().sum())
    protected_prop = protected_accepted / protected_total
    print(protected.title() + ' proportion accepted: %.3f' % protected_prop)

    # return adverse impact ratio
    return protected_prop / reference_prop


def marginal_effect(cm_dict, reference, protected):
    """ Calculates the marginal effect as a percentage difference between a reference and
        a protected group: reference_percent - protected_percent. Prints intermediate values.
        Tightly coupled to cm_dict.

        :param cm_dict: Dict of confusion matrices containing information
                        about reference and protected groups.
        :param reference: Name of reference group in cm_dict as a string.
        :param protected: Name of protected group in cm_dict as a string.
        :return: Marginal effect value.

    """

    # reference group summary
    reference_accepted = float(cm_dict[reference].iat[1, 0] + cm_dict[reference].iat[1, 1])  # predicted 0's
    reference_total = float(cm_dict[reference].sum().sum())
    reference_percent = 100 * (reference_accepted / reference_total)
    print(reference.title() + ' accepted: %.2f%%' % reference_percent)

    # protected group summary
    protected_accepted = float(cm_dict[protected].iat[1, 0] + cm_dict[protected].iat[1, 1])  # predicted 0's
    protected_total = float(cm_dict[protected].sum().sum())
    protected_percent = 100 * (protected_accepted / protected_total)
    print(protected.title() + ' accepted: %.2f%%' % protected_percent)

    # return marginal effect
    return reference_percent - protected_percent


def smd(valid, x_name, yhat_name, reference, protected):
    """ Calculates standardized mean difference between a protected and reference group:
        (mean(yhat | x_j=protected) - mean(yhat | x_j=reference))/sigma(yhat).
        Prints intermediate values.

        :param valid: Pandas dataframe containing j and predicted (yhat) values.
        :param x_name: name of demographic column containing reference and protected group labels.
        :param yhat_name: Name of predicted value column.
        :param reference: name of reference group in x_name.
        :param protected: name of protected group in x_name.

    Returns:
       Standardized mean difference as a formatted string.

    """

    # yhat mean for j=reference
    reference_yhat_mean = valid[valid[x_name] == reference][yhat_name].mean()
    print(reference.title() + ' mean yhat: %.2f' % reference_yhat_mean)

    # yhat mean for j=protected
    protected_yhat_mean = valid[valid[x_name] == protected][yhat_name].mean()
    print(protected.title() + ' mean yhat: %.2f' % protected_yhat_mean)

    # std for yhat
    sigma = valid[yhat_name].std()
    print(yhat_name.title() + ' std. dev.:  %.2f' % sigma)

    return (protected_yhat_mean - reference_yhat_mean) / sigma