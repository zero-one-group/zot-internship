import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.optimize as opt 
from scipy.optimize import minimize 
from sklearn.preprocessing import PolynomialFeatures


def clean_data(file_path, variables): 
    data = pd.read_csv(file_path, header=None)
    data.columns = variables
    return data

def extract_features(data): 
    x = data.iloc[:, :-1]
    number_of_samples = len(x) 
    intercept_variable = np.ones(number_of_samples) 
    x = np.c_[intercept_variable, x]

    y = data.iloc[:,2]
    y = y[:, np.newaxis]
    return x, y

def sigmoid(z): 
    return 1 / (1+np.exp(-z))

def compute_cost(theta, x, y):
    number_of_samples = len(x)
    predictions = sigmoid(x @ theta)

    negative_cost = np.log(predictions).T @ y
    positive_cost = np.log(1-predictions).T @ (1-y)
    cost = (-1/number_of_samples) * (negative_cost + positive_cost)
    return sum(cost)

def compute_gradient(theta, x, y):
    number_of_samples = len(y) 
    predictions = sigmoid(x @ theta.reshape(-1, 1))

    delta = x.T @ (predictions-y) 
    gradient = (1/number_of_samples) * delta 
    return gradient.flatten()

def optimised_cost(theta, x, y): 
    return minimize(compute_cost, 
                    theta, 
                    args=(x, y), 
                    method=None, 
                    jac=compute_gradient, 
                    options={'maxiter':400})

def accuracy(theta, x, y, threshold): 
    probability = sigmoid(x.dot(theta.T)) >= threshold
    probability = probability.astype('int') 
    return 100 * sum(probability == y.ravel())/probability.size

def plot_data(data, xlabel, ylabel): 
    def get_x(data): 
        return data.iloc[:, 0]

    def get_y(data): 
        return data.iloc[:, 1]

    admitted = data[data.iloc[:, -1] == 1]
    not_admitted = data[data.iloc[:, -1] == 0]
    plt.scatter(get_x(admitted), get_y(admitted), marker='+', color='black', label='passed')
    plt.scatter(get_x(not_admitted), get_y(not_admitted), marker='o', color='yellow', edgecolor='black', label='failed admitted') 
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('figure 1: scatter plot of training data')

    return plt.show() 

def insert_polynomial_features(x): 
    x = x[:, 1:3]
    polynomial_features = PolynomialFeatures(6)
    return polynomial_features.fit_transform(x)

def compute_regularised_cost(theta, lambda_value, x, y): 
    m = y.size
    h = sigmoid(x.dot(theta))
    
    negative_cost = np.log(h).T.dot(y)
    positive_cost = np.log(1-h).T.dot(1-y)
    regularisation = (lambda_value/(2*m))*np.sum(np.square(theta[1:]))

    cost = (-1/m) * (negative_cost + positive_cost) + regularisation
    return sum(cost)

def compute_regularised_gradient(theta, lambda_value, x, y):
    number_of_samples = y.size
    predictions = sigmoid(x @ theta.reshape(-1,1))

    intial_gradient = (1/number_of_samples) * x.T @ (h-y) 
    regularisation = (lambda_value/number_of_samples) * np.r_[[[0]], theta[1:].reshape(-1,1)]
    gradient = initial_gradient + regularisation
    return gradient.flatten()


if __name__ == '__main__':
    # PART 1
    file_path = 'data/ex2data1.txt'
    variables = ['score_1', 'score_2', 'admitted']
    exam_data = clean_data(file_path, variables)

    x_label = ('exam_score_1')
    y_label = ('exam_score_2')
    plot_data(exam_data, x_label, y_label)

    x, y = extract_features(exam_data)
    number_of_samples, features = x.shape
    theta = np.zeros(features)

    cost = compute_cost(theta, x, y)
    optimised_cost_function = optimised_cost(theta, x, y)

    threshold = 0.5
    exam_score_samples = np.array([1, 45, 85])
    optimised_theta = optimised_cost_function.x
    sigmoid(np.array(exam_score_samples).dot(optimised_theta.T))
    accuracy = accuracy(optimised_theta, x, y, threshold)


    # PART 2 
    file_path = 'data/ex2data2.txt' 
    variables = ['test_2', 'test_2', 'pass']
    microchip_data = clean_data(file_path, variables)

    xlabel = ('microchip test 1')
    ylabel = ('microchip test 2')
    plot_data(microchip_data, xlabel, ylabel)

    x, y = extract_features(microchip_data)
    x = insert_polynomial_features(x)
    m, n = x.shape
    theta = np.zeros(n)
    lambda_value = 1

    regularised_cost = compute_regularised_cost(theta, lambda_value, x, y)
