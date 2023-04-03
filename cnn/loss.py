import numpy as np

def MSE(y_actual, y_pred):
    return np.mean(np.power((y_actual - y_pred), 2))

def MSE_prime(y_actual, y_pred):
    return -2 * (y_actual - y_pred)/y_actual.size

def binary_cross_entropy(y_actual, y_pred):
    return -np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_actual, y_pred):
    return ((1/np.size(y_actual)) * (((1 - y_actual)/(1 - y_pred)) - (y_actual/y_pred)))