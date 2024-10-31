import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from adsp31014 import FTest, format_categorical_predictors, observation_leverage, simple_residual, \
    standardized_residual, deleted_residual, studentized_residual
from adsp31014 import backward_selection, linear_regression
from adsp31014 import box_cox, inverse_box_cox

IDEAL_POWER = -0.5


# Constants for Markdown table formatting
DELIMITER = "|"
HEADER_SEPARATOR = "-"
NUMERIC_FORMAT = "{:.4g}"

def get_outliers_from_residuals(simple_residuals, standardized_residuals, deleted_residuals, studentized_residuals):
    def detect_outliers(residuals):
        Q1 = residuals.quantile(0.25)
        Q3 = residuals.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = residuals[(residuals < lower_bound) | (residuals > upper_bound)]
        return set(outliers.index)

    # Detect outliers for each type of residuals
    simple_outliers_idx = detect_outliers(simple_residuals)
    standardized_outliers_idx = detect_outliers(standardized_residuals)
    deleted_outliers_idx = detect_outliers(deleted_residuals)
    studentized_outliers_idx = detect_outliers(studentized_residuals)

    # Combine all outliers into a single set
    all_outliers_idx = simple_outliers_idx | standardized_outliers_idx | deleted_outliers_idx | studentized_outliers_idx

    return list(all_outliers_idx)


def reverse_get_dummies(df, sep: str = '_'):
    # Detecting all prefixes from dummy columns
    dummy_groups = {}
    for col in df.columns:
        if sep in col:
            prefix = col.split(sep)[0]
            if prefix not in dummy_groups:
                dummy_groups[prefix] = []
            dummy_groups[prefix].append(col)

    # Process each group to revert to the original categorical column
    for prefix, dummy_columns in dummy_groups.items():
        # Extracting the original categorical values from the column names
        categories = {col: col.split(sep, 1)[1] for col in dummy_columns}
        # Determining the new column values
        new_column = df[dummy_columns].idxmax(axis=1).map(categories)
        # Dropping the dummy columns from the DataFrame
        df = df.drop(columns=dummy_columns)
        # Adding the new categorical column
        df[prefix] = new_column

    return df

def format_cell(value, width) -> str:
    formatted_value = NUMERIC_FORMAT.format(value) if isinstance(value, float) else str(value)
    return formatted_value.ljust(width)


def format_row(row, column_widths) -> str:
    formatted_cells = [format_cell(value, width) for value, width in zip(row, column_widths)]
    return DELIMITER + " " + " | ".join(formatted_cells) + " " + DELIMITER


def create_separator(columns) -> str:
    return DELIMITER + " " + " | ".join([HEADER_SEPARATOR * len(col) for col in columns]) + " " + DELIMITER


def convert_dataframe_to_markdown(df) -> str:
    columns = df.columns
    column_widths = [max(len(str(col)), 4) for col in columns]  # Ensure a minimum width of 4
    header = DELIMITER + " " + " | ".join(
        col.ljust(width) for col, width in zip(columns, column_widths)) + " " + DELIMITER
    separator = create_separator(columns)
    rows = [format_row(row, column_widths) for _, row in df.iterrows()]
    return "\n".join([header, separator] + rows)


def print_dataframe_as_markdown(df) -> None:
    markdown_table = convert_dataframe_to_markdown(df)
    print(markdown_table)

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

def display_residuals(simple, standardized, deleted, studentized):
    diagnostic_df = pd.DataFrame({
                                      'Simple Residual': simple,
                                      'Standardized Residual': standardized,
                                      'Deleted Residual': deleted,
                                      'Studentized Residual': studentized})

    # Identify Influential Observations
    stats = diagnostic_df.columns

    fig, axs = plt.subplots(1, 4, figsize=(14, 8), dpi=200)
    plt.subplots_adjust(wspace=0.25)
    for j in range(4):
        col = stats[j]
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

    result = linear_regression(selected_predictors, target)
    predictions = selected_predictors.dot(result.parameter_table['Estimate'])
    predictions.name = "Prediction"

    leverage = observation_leverage(selected_predictors, result)
    leverage.name = "Leverage"

    simple = simple_residual(target, predictions)
    standardized = standardized_residual(simple, result.residual_variance, leverage)
    deleted = deleted_residual(simple, leverage)
    studentized = studentized_residual(simple, leverage, target, selected_predictors)

    display_leverage(leverage, len(selected_predictors.columns), len(selected_predictors))
    display_residuals(simple, standardized, deleted, studentized)

    result_df = df['Sale Price'].to_frame().join(inverse_box_cox(predictions, IDEAL_POWER)).join(reverse_get_dummies(selected_predictors))
    leverage_df = pd.DataFrame(leverage, columns=['Leverage']).join(result_df)

    print("\n High Leverage Observations")
    high_leverage = leverage_df[leverage_df['Leverage'] > 0.15]

    print_dataframe_as_markdown(high_leverage)

    outliers = get_outliers_from_residuals(simple, standardized, deleted, studentized)
    outliers_df = result_df.loc[outliers]

    print("\n Outlier Observations")
    print_dataframe_as_markdown(outliers_df)



