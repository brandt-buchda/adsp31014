import pandas as pd
import matplotlib.pyplot as plt

from adsp31014 import *
from data.twenty_chicago_taxi_trip import clean_and_format_data

def test_backwards_selection():
    data = pd.read_csv('../data/Twenty_Chicago_Taxi_Trip.csv')

    backward_selection(data['Trip_Payment'].to_frame(), data[['Trip_Minutes', 'Trip_Miles']], data['Payment_Method'].to_frame(), debug=True)

def test_high_leverage_observations():
    data = pd.read_csv('../data/Twenty_Chicago_Taxi_Trip.csv')
    X, y = clean_and_format_data(data)

    result = linear_regression(X, y)
    prediction =  X.dot(result.parameter_table['Estimate'])

    leverage = observation_leverage(X, result)

    simple = simple_residual(y, prediction)
    standardized = standardized_residual(simple, result.residual_variance, leverage)
    deleted = deleted_residual(simple, leverage)
    studentized = studentized_residual(simple, leverage, y, X)

    diagnostic_df = pd.DataFrame({'Response': y,
                                      'Prediction': prediction,
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

    pass

def test_outlier_observations():

    pass

# Run tests
#test_backwards_selection()
test_high_leverage_observations()
test_outlier_observations()