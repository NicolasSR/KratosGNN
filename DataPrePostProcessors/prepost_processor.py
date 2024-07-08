import numpy as np
from sklearn import preprocessing

def scaler_functions(scaler_name):
    if scaler_name == "minmax":
        sc_fun = preprocessing.MinMaxScaler()
    elif scaler_name == "robust":
        sc_fun = preprocessing.RobustScaler()
    elif scaler_name == "standard":
        sc_fun = preprocessing.StandardScaler()
    else:
        raise Exception('No valid scaler name was selected')
    return sc_fun