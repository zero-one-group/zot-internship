import numpy as np
import pandas as pd
from scipy.io import loadmat

# LOADING FILES 
def load_mat(file_path):
    """return mat file"""
    return loadmat(file_path)

def load_csv(file_path):
    """return csv file"""
    return pd.read_csv(file_path) 

def extract_variables(data):
    """return variables"""
    x = data['X']
    m = len(x)
    ones = np.ones(m)
    x = np.c_[ones, x]

    y = data['y']

    return x, y

def extract_x(data):
    return data['X']

# FORMULAS
def h(x, theta): 
    """return h"""
    return np.dot(x, theta)

def sigmoid(z):
    """return sigmoid function"""
    return 1 / (1+np.exp(-z))

def cost_function(theta, x, y):
    """return cost function"""
    m = len(x)
    h = sigmoid(x.dot(theta))

    negative_cost = np.log(h).T.dot(y)
    positive_cost = np.log(1-h).T.dot(1-y)
    j = (-1/m) * (negative_cost + positive_cost)

    return np.float(j[0])

def regularised_cost_function(theta, lambda_value, x, y): 
    """return regularised cost"""
    m = y.size
    h = sigmoid(x.dot(theta))
    
    negative_cost = np.log(h).T.dot(y)
    positive_cost = np.log(1-h).T.dot(1-y)
    regularisation = (lambda_value/(2*m))*np.sum(np.square(theta[1:]))

    j = (-1/m) * (negative_cost + positive_cost) + regularisation
    return j[0]

def regularised_gradient(theta, lambda_value, x, y):
    """return regularised gradient"""
    m = y.size
    h = sigmoid(x.dot(theta.reshape(-1,1)))

    initial_gradient = (1/m) * x.T.dot(h-y)
    regularisation = (lambda_value/m) * np.r_[[[0]], theta[1:].reshape(-1,1)]

    gradient = initial_gradient + regularisation
    return gradient.flatten()

# COMPUTE ACCURACIES 
def compute_accuracy(theta_values, x, y):
    """return accuracy"""
    p = sigmoid(x.dot(theta_values.T))
    p = np.argmax(p, axis=1) + 1
    return np.mean(p == y.ravel()) * 100
