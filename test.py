import pandas as pd
from adsp31014 import *

def test_backwards_selection():
    data = pd.read_csv('./data/Twenty_Chicago_Taxi_Trip.csv')

    backward_selection(data['Trip_Payment'].to_frame(), data[['Trip_Minutes', 'Trip_Miles']], data['Payment_Method'].to_frame(), DEBUG=True)


test_backwards_selection()
