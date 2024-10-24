"""
Name: Week 2 Chicago Taxi Trip Regression Backward.py
Course: ADSP 31014 Statistical Models for Data Science
Author: Ming-Long Lam, Ph.D.
Organization: University of Chicago
Last Modified: October 8, 2024
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.stats import f

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10f}'.format

#sys.path.append('C:\\ADSP\\ADSP31014\\Code')

import Regression

# The FSig is the sixth element in each row of the FTest
def takeFSig(s):
    return s[6]

TaxiTrip = pandas.read_csv('../data/Twenty_Chicago_Taxi_Trip.csv')

cat_name = ['Payment_Method']
int_name = ['Trip_Minutes', 'Trip_Miles']

candidate_name = cat_name + int_name
candidate_count = len(candidate_name)

target_name = 'Trip_Payment'

train_data = TaxiTrip[candidate_name + [target_name]].dropna().reset_index(drop = True)

n_sample = train_data.shape[0]
y = train_data[target_name]

remove_threshold = 0.05
q_show_diary = True
step_diary = []

var_in_model = ['Intercept'] + candidate_name

# Step 0: Enter Intercept and all candidates
X1 = pandas.get_dummies(train_data[cat_name].astype('category'), dtype = float)
X1 = X1.join(train_data[int_name])
X1.insert(0, 'Intercept', 1.0)
X1_columns = X1.columns

result_list = Regression.LinearRegression(X1, y)
m1 = len(result_list[5])

# residual_variance = result_list[2] and residual_df = result_list[3]
SSE1 = result_list[2] * result_list[3]

step_diary.append([0, 'None', SSE1, m1] + 4 * [numpy.nan])

# Backward Selection Steps
for iStep in range(candidate_count):
    FTest = []
    for pred in candidate_name:
        drop_cols = [col for col in X1_columns if pred in col]
        X = X1.drop(columns = drop_cols)

        result_list = Regression.LinearRegression(X, y)
        m0 = len(result_list[5])
        SSE0 = result_list[2] * result_list[3]

        df_numer = m1 - m0
        df_denom = n_sample - m1
        if (df_numer > 0 and df_denom > 0):
            FStat = ((SSE0 - SSE1) / df_numer) / (SSE1 / df_denom)
            FSig = f.sf(FStat, df_numer, df_denom)
            FTest.append([pred, SSE0, m0, FStat, df_numer, df_denom, FSig])

    # Show F Test results for the current step
    if (q_show_diary): 
        print('\n===== F Test Results for the Current Backward Step =====')
        print('Step Number: ', iStep)
        print('Step Diary:')
        print('[Variable Candidate | Residual Sum of Squares | N Non-Aliased Parameters | F Stat | F DF1 | F DF2 | F Sig]')
        for row in FTest:
            print(row)

    FTest.sort(key = takeFSig, reverse = True)
    FSig = takeFSig(FTest[0])
    if (FSig >= remove_threshold):
        remove_var = FTest[0][0]
        SSE1 = FTest[0][1]
        m1 = FTest[0][2]
        step_diary.append([iStep+1] + FTest[0])
        drop_cols = [col for col in X1_columns if remove_var in col]
        X1 = X1.drop(columns = drop_cols)
        X1_columns = X1.columns
        var_in_model.remove(remove_var)
        candidate_name.remove(remove_var)
    else:
        break

backward_summary = pandas.DataFrame(step_diary, columns = ['Step', 'Variable Removed', 'Residual Sum of Squares', 'N Non-Aliased Parameters', 'F Stat', 'F DF1', 'F DF2', 'F Sig'])
