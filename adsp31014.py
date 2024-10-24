import numpy as np
import pandas as pd
from scipy.stats import t, f

class LinearRegressionResults:
    def __init__(self, parameter_table, covariance_matrix, residual_variance, residual_degree_freedom, aliased_parameters, non_aliased_parameters):
        self.parameter_table = parameter_table
        self.covariance_matrix = covariance_matrix
        self.residual_variance = residual_variance
        self.residual_degree_freedom = residual_degree_freedom
        self.aliased_parameters = aliased_parameters
        self.non_aliased_parameters = non_aliased_parameters
    pass


class FTest:
    def __init__(self, predictor, sse, size, f_stat, f_sig, num_removed_step, num_removed_total):
        self.predictor = predictor
        self.sse = sse
        self.size = size
        self.f_stat = f_stat
        self.f_sig = f_sig
        self.num_removed_step = num_removed_step
        self.num_removed_total = num_removed_total

    @staticmethod
    def print_header():
        print(f"{'-' * 96}")
        print(f"{'Predictor':<15}{'SSE':<10}{'Size':<10}{'F-Statistic':<15}{'F-Significance':<17}{'Removed Step':<15}{'Removed Total':<15}")
        print(f"{'-' * 96}")


    def print(self):
        print(f"{self.predictor:<15}{self.sse:<10.4f}{self.size:<10}{self.f_stat:<15.4f}{self.f_sig:<17.4e}{self.num_removed_step:<15}{self.num_removed_total:<15}")



def sweep_operator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    ''' Implement the SWEEP operator

    Parameter
    ---------
    pDim: dimension of matrix inputM, integer greater than one
    inputM: a square and symmetric matrix, numpy array
    origDiag: the original diagonal elements before any SWEEPing
    sweepCol: a list of columns numbers to SWEEP
    tol: singularity tolerance, positive real

    Return
    ------
    A: negative of a generalized inverse of input matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = np.copy(inputM)
    ANext = np.zeros((pDim ,pDim))

    for k in sweepCol:
        Akk = A[k ,k]
        pivot = tol * abs(origDiag[k])
        if (not np.isinf(Akk) and abs(Akk) >= pivot and pivot > 0.0):
            nonAliasParam.append(k)
            ANext = A - np.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / abs(Akk)
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
        A = ANext
    return (A, aliasParam, nonAliasParam)


def linear_regression (X, y, tolSweep = 1e-7):
    ''' Train a linear regression model

    Argument
    --------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    A list of model output:
    (0) parameter_table: a Pandas DataFrame of regression coefficients and statistics
    (1) cov_matrix: a Pandas DataFrame of covariance matrix for regression coefficient
    (2) residual_variance: residual variance
    (3) residual_df: residual degree of freedom
    (4) aliasParam: a list of aliased rows/columns in input matrix
    (5) nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    # X: A Pandas DataFrame, rows are observations, columns are regressors
    # y: A Pandas Series, rows are observations of the response variable

    Z = X.join(y)
    n_sample = X.shape[0]
    n_param = X.shape[1]

    ZtZ = Z.transpose().dot(Z)
    diag_ZtZ = np.diagonal(ZtZ)
    eps_double = np.finfo(np.float64).eps
    tol = np.sqrt(eps_double)

    ZtZ_transf, aliasParam, nonAliasParam = sweep_operator ((n_param +1), ZtZ, diag_ZtZ, sweepCol = range(n_param), tol = tol)

    residual_df = n_sample - len(nonAliasParam)
    residual_variance = ZtZ_transf[n_param, n_param] / residual_df

    b = ZtZ_transf[0:n_param, n_param]
    b[aliasParam] = 0.0

    parameter_name = X.columns

    XtX_Ginv = - residual_variance * ZtZ_transf[0:n_param, 0:n_param]
    XtX_Ginv[:, aliasParam] = 0.0
    XtX_Ginv[aliasParam, :] = 0.0
    cov_matrix = pd.DataFrame(XtX_Ginv, index = parameter_name, columns = parameter_name)

    parameter_table = pd.DataFrame(index = parameter_name,
                                   columns = ['Estimate' ,'Standard Error', 't', 'Significance', 'Lower 95 CI', 'Upper 95 CI'])
    parameter_table['Estimate'] = b
    parameter_table['Standard Error'] = np.sqrt(np.diag(cov_matrix))
    parameter_table['t'] = np.divide(parameter_table['Estimate'], parameter_table['Standard Error'])
    parameter_table['Significance'] = 2.0 * t.sf(abs(parameter_table['t']), residual_df)

    t_critical = t.ppf(0.975, residual_df)
    parameter_table['Lower 95 CI'] =  parameter_table['Estimate'] - t_critical * parameter_table['Standard Error']
    parameter_table['Upper 95 CI'] =  parameter_table['Estimate'] + t_critical * parameter_table['Standard Error']

    return LinearRegressionResults(parameter_table, cov_matrix, residual_variance, residual_df, aliasParam, nonAliasParam)

def box_cox(target, power) -> pd.Series:
    return np.log(target) if power == 0 else (np.power(target, power) - 1.0) / power

def inverse_box_cox(target, power) -> pd.Series:
    return np.exp(target) if power == 0 else np.power(target * power + 1, 1 / power)

def format_categorical_predictors(predictors: pd.DataFrame):
    #TODO: Order categorical predictors in ascending order by number of observations
    #TODO: Central AC is binary, should it be encoded as 2 columns?
    return pd.get_dummies(predictors.astype('category'), dtype=float)

def backward_selection(target, continuous_predictors, categorical_predictors, threshold = 0.05, debug=False):
    original_predictor_labels = np.concatenate((categorical_predictors.columns.values, continuous_predictors.columns.values)).tolist()
    predictors = format_categorical_predictors(categorical_predictors).join(continuous_predictors)
    predictor_labels = predictors.columns.values
    predictors.dropna(inplace=True)

    num_observations = len(predictors)

    # Initial regression fit
    predictors.insert(0, 'Intercept', 1.0)

    results: LinearRegressionResults = linear_regression(predictors, target)
    outer_model_size = len(results.non_aliased_parameters)
    outer_sse = results.residual_variance * results.residual_degree_freedom # Sum of Square Error


    history = [{"step": 0, "test": FTest('None', outer_sse, outer_model_size, np.nan, np.nan, np.nan, np.nan) }]

    for step in range(len(original_predictor_labels)):
        f_tests = []

        # Test everything except Intercept
        for predictor in original_predictor_labels:
            # Drop all columns for categorical columns
            drop_cols = [col for col in predictor_labels if predictor in col]
            test_predictors = predictors.drop(columns=drop_cols)

            result = linear_regression(test_predictors, target)
            inner_model_size = len(result.non_aliased_parameters)
            inner_sse = result.residual_variance * result.residual_degree_freedom

            f_d1 = outer_model_size - inner_model_size
            f_d2 = num_observations - outer_model_size

            if f_d1 > 0 and f_d2 > 0:
                f_stat = ((inner_sse - outer_sse) / f_d1) / (outer_sse / f_d2)
                f_sig = f.sf(f_stat, f_d1, f_d2)
                f_tests.append(FTest(predictor, inner_sse, inner_model_size, f_stat, f_sig, f_d1, f_d2))

        if debug:
            print('\n===== F Test Results for the Current Backward Step =====')
            print('Step Number: ', step)

            FTest.print_header()
            for row in f_tests:
               row.print()

        f_tests.sort(key=lambda x: x.f_sig, reverse=True)
        worst_predictor = f_tests[0]

        if worst_predictor.f_sig >= threshold:
            remove = worst_predictor.predictor
            outer_sse = worst_predictor.sse
            outer_model_size = worst_predictor.size

            history.append({"step": step + 1, "test": worst_predictor})

            drop_cols = [col for col in predictor_labels if remove in col]
            predictors = predictors.drop(columns=drop_cols)
            original_predictor_labels.remove(worst_predictor.predictor)
            predictor_labels = predictors.columns

        else:
            break

    return history