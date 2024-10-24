import pandas as pd
from adsp31014 import *
from data.twenty_chicago_taxi_trip import clean_and_format_data

def test_backwards_selection():
    data = pd.read_csv('../data/Twenty_Chicago_Taxi_Trip.csv')

    backward_selection(data['Trip_Payment'].to_frame(), data[['Trip_Minutes', 'Trip_Miles']], data['Payment_Method'].to_frame(), debug=True)

def test_high_leverage_observations():
    data = pd.read_csv('../data/Twenty_Chicago_Taxi_Trip.csv')
    X, y = clean_and_format_data(data)

    result = linear_regression(X, y)

    leverage = observation_leverage(X, result)

    pass

def test_outlier_observations():

    pass

# Run tests
#test_backwards_selection()
test_high_leverage_observations()
test_outlier_observations()