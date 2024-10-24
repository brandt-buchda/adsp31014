import pandas as pd
import numpy as np
from pandas.io.pytables import attribute_conflict_doc

import Regression
from scipy.stats import f

from adsp31014 import LinearRegressionResults, FTest
from adsp31014 import linear_regression
from adsp31014 import backward_selection

IDEAL_POWER = -0.5

def box_cox(target, power) -> pd.Series:
    return np.log(target) if power == 0 else (np.power(target, power) - 1.0) / power

def inverse_box_cox(target, power) -> pd.Series:
    return np.exp(target) if power == 0 else np.power(target * power + 1, 1 / power)

def format_categorical_predictors(predictors: pd.DataFrame):
    #TODO: Order categorical predictors in ascending order by number of observations
    #TODO: Central AC is binary, should it be encoded as 2 columns?
    return pd.get_dummies(predictors, drop_first=True, dummy_na=True, dtype=float)

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

    target = box_cox(df['Sale Price'], IDEAL_POWER).to_frame(name='BC Sale Price')
    categorical_predictors = df[categorical_predictor_labels]
    continuous_predictors = df[continuous_predictor_labels]

    selection = backward_selection(target, continuous_predictors, categorical_predictors)

    print(selection)

    pass