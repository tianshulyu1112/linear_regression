### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

### my test
def simulate_data(n=1000):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    x0 = np.ones(n)
    x1 = np.random.exponential(9000,n)
    x2 = np.random.poisson(15,n)
    residual = np.random.normal(0,1,n)
    beta = np.random.normal(0,2.5,size=np.transpose(np.array([x0,x1,x2])).shape[1])
    data={}
    data['beta'] = beta
    data['X'] = np.transpose(np.array([x0,x1,x2]))
    data['y'] = data['X'].dot(beta) + residual
    return data

def compare_models():
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    pass


def load_hospital_data(path_to_data):
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    clean_df = pd.read_csv(path_to_data)
    return clean_df


def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """
    pass


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###