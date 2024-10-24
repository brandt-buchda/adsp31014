import pandas as pd
import numpy as np

from adsp31014 import FTest
from adsp31014 import backward_selection, linear_regression
from adsp31014 import box_cox, inverse_box_cox

IDEAL_POWER = -0.5

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

    selected, removed = backward_selection(target, continuous_predictors, categorical_predictors)

    print("Selected Features:")
    FTest.print_header()
    for row in selected:
        row.print()

    print("\nRemoved Features:")
    FTest.print_header()
    for row in removed:
        row.print()

