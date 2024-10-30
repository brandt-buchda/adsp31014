import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from adsp31014 import FTest, format_categorical_predictors, observation_leverage, simple_residual, \
    standardized_residual, deleted_residual, studentized_residual
from adsp31014 import backward_selection, linear_regression
from adsp31014 import box_cox, inverse_box_cox

IDEAL_POWER = -0.5

def display_leverage(leverage, num_features, num_observations, bin_width=0.05):
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, dpi=200, sharex=True,
                                   figsize=(12, 6),
                                   gridspec_kw={'height_ratios': [4, 1]})

    ax0.hist(leverage, color='royalblue', bins=np.arange(min(leverage), max(leverage) + bin_width, bin_width))
    ax0.axvline(num_features / num_observations, color='orangered', linestyle=':')
    ax0.set_xlabel('')
    ax0.set_ylabel('Number of Observations')
    ax0.yaxis.grid(True)

    ax1.boxplot(leverage, vert=False)
    ax1.axvline(num_features / num_observations, color='orangered', linestyle=':')
    ax1.set_xlabel('Leverage')
    ax1.set_ylabel('')
    ax1.xaxis.set_major_locator(MultipleLocator(base=0.05))
    ax1.xaxis.set_minor_locator(MultipleLocator(base=0.01))
    ax1.yaxis.set_major_locator(MultipleLocator(base=1.0))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.grid(True)
    plt.show()

    pass

def display_residuals(target, predictors, result, leverage):

    predictions = predictors.dot(result.parameter_table['Estimate'])

    simple = simple_residual(target, predictions)
    standardized = standardized_residual(simple, result.residual_variance, leverage)
    deleted = deleted_residual(simple, leverage)
    studentized = studentized_residual(simple, leverage, target, predictors)

    diagnostic_df = pd.DataFrame({'Response': target,
                                      'Prediction': predictions,
                                      'Leverage': leverage,
                                      'Simple Residual': simple,
                                      'Standardized Residual': standardized,
                                      'Deleted Residual': deleted,
                                      'Studentized Residual': studentized})

    # Identify Influential Observations
    stats = diagnostic_df.columns

    fig, axs = plt.subplots(1, 4, figsize=(12, 8), sharey=True, dpi=200)
    plt.subplots_adjust(wspace=0.15)
    for j in range(4):
        col = stats[j + 3]
        ax = axs[j]
        ax.boxplot(diagnostic_df[col], vert=True)
        ax.axhline(0.0, color='red', linestyle=':')
        ax.set_xticks([])
        ax.set_xlabel(col)
        ax.grid(axis='y')
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('./data/NorthChicagoTownshipHomeSale.csv')

    categorical_predictor_labels = [
        'Wall Material',
        'Roof Material',
        'Basement',
        'Central Air Conditioning']

    continuous_predictor_labels = [
        'Age',
        'Bedrooms',
        'Building Square Feet',
        'Full Baths',
        'Garage Size',
        'Half Baths',
        'Land Acre',
        'Tract Median Income'
    ]

    target = box_cox(df['Sale Price'], IDEAL_POWER).to_frame(name='BC Sale Price').squeeze()
    categorical_predictors = df[categorical_predictor_labels]
    continuous_predictors = df[continuous_predictor_labels]

    selected, removed = backward_selection(target, continuous_predictors, categorical_predictors)

    selected_predictors = [element.predictor for element in selected]

    print("Selected Features:")
    FTest.print_header()
    for row in selected:
        row.print()

    print("\nRemoved Features:")
    FTest.print_header()
    for row in removed:
        row.print()

    selected_categorical_predictors = list(set(categorical_predictor_labels).intersection(selected_predictors))
    selected_categorical_predictors = format_categorical_predictors(df[selected_categorical_predictors])
    selected_continuous_predictors = list(set(continuous_predictor_labels).intersection(selected_predictors))
    selected_continuous_predictors = df[selected_continuous_predictors]

    selected_predictors = selected_categorical_predictors.join(selected_continuous_predictors)
    selected_predictors.insert(0, 'Intercept', 1.0)

    results = linear_regression(selected_predictors, target)

    leverage = observation_leverage(selected_predictors, results)

    display_leverage(leverage, len(selected_predictors.columns), len(selected_predictors))

