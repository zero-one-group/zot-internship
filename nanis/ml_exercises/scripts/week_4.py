import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def extract_variables(data):
    x = data['X']
    number_of_samples = len(x)
    ones = np.ones(number_of_samples)
    x = np.c_[ones, x]

    y = data['y']

    return x, y

def visualise_sample(x):    
    number_of_samples = len(y)
    sample_columns = np.random.choice(number_of_samples, 20)
    plt.imshow(x[sample_columns, 1:].reshape(-1, 20).T)
    plt.axis('off')

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def regularised_cost_function(theta, lambda_value, x, y): 
    number_of_samples = y.size
    predictions = sigmoid(x.dot(theta))
    
    negative_cost = np.log(predictions).T.dot(y)
    positive_cost = np.log(1-predictions).T.dot(1-y)
    regularisation = (lambda_value/(2*number_of_samples))*np.sum(np.square(theta[1:]))

    cost = (-1/number_of_samples) * (negative_cost + positive_cost) + regularisation
    return cost[0]

def regularised_gradient(theta, lambda_value, x, y):
    number_of_samples = y.size
    predictions = sigmoid(x.dot(theta.reshape(-1,1)))

    initial_gradient = (1/number_of_samples) * x.T.dot(predictions-y)
    regularisation = (lambda_value/number_of_samples) * np.r_[[[0]], theta[1:].reshape(-1,1)]

    gradient = initial_gradient + regularisation
    return gradient.flatten()

def one_vs_all(x, y, n_labels, lambda_value): 
    number_of_samples, features = x.shape
    input_theta = np.zeros((features, 1))
    theta_values = np.zeros((n_labels, features))

    for classes in np.arange(1, n_labels+1): 
        optimised_cost_function = minimize(regularised_cost_function, 
                                           input_theta, 
                                           args=(lambda_value, x, (y == classes)*1),
                                           method=None,
                                           jac=regularised_gradient,
                                           options={'maxiter':50})
        theta_values[classes-1] = optimised_cost_function.x
    return theta_values

def compute_accuracy(theta_values, x, y):
    probability = sigmoid(x.dot(theta_values.T))
    probability = np.argmax(probability, axis=1) + 1
    return np.mean(probability == y.ravel()) * 100

def multiclass_logistic_regression(x, y):
    classifier = LogisticRegression(C=10, penalty='l2', solver='liblinear')
    classifier.fit(x[:,1:], y.ravel())

    prediction = classifier.predict(x[:, 1:])
    return np.mean(prediction == y.ravel()) * 100

def neural_networks(theta_1, theta_2, x): 
    number_of_samples, features = x.shape

    z2 = theta_1.dot(x.T)
    a2 = np.c_[np.ones((number_of_samples,1)), sigmoid(z2).T]

    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)

    prediction = np.argmax(a3, axis=1) + 1
    return np.mean(prediction == y.ravel()) * 100


if __name__ == '__main__': 
    file_path = 'data/ex3data1.mat'
    digits_data = loadmat(file_path) 
    x, y = extract_variables(digits_data)

    visualise_sample(x) 

    n_labels = 10
    lambda_value = 0.1
    theta_values = one_vs_all(x, y, n_labels, lambda_value)
    accuracy = compute_accuracy(theta_values, x, y)

    multiclass_logistic_regression_accuracy = multiclass_logistic_regression(x, y)

    weights = loadmat('data/ex3weights.mat')
    theta_1, theta_2 = weights['Theta1'], weights['Theta2']
    neural_networks_accuracy = neural_networks(theta_1, theta_2, x)
