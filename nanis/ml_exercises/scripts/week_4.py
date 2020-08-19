import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def extract_variables(data):
    inputs = data['X']
    number_of_samples = len(inputs)
    ones = np.ones(number_of_samples)
    inputs = np.c_[ones, inputs]

    outputs = data['y']

    return inputs, outputs

def visualise_sample(inputs):    
    number_of_samples = len(outputs)
    sample_columns = np.random.choice(number_of_samples, 20)
    plt.imshow(inputs[sample_columns, 1:].reshape(-1, 20).T)
    plt.axis('off')

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def regularised_cost_function(theta, lambda_value, inputs, outputs): 
    number_of_samples = outputs.size
    predictions = sigmoid(inputs @ theta)
    
    negative_cost = np.log(predictions).T @ (outputs)
    positive_cost = np.log(1-predictions).T @ (1-outputs)
    regularisation = (lambda_value/(2*number_of_samples))*np.sum(np.square(theta[1:]))

    cost = (-1/number_of_samples) * (negative_cost + positive_cost) + regularisation
    return sum(cost)

def regularised_gradient(theta, lambda_value, inputs, outputs):
    number_of_samples = outputs.size
    predictions = sigmoid(inputs @ (theta.reshape(-1,1)))

    initial_gradient = (1/number_of_samples) * inputs.T @ (predictions-outputs)
    regularisation = (lambda_value/number_of_samples) * np.r_[[[0]], theta[1:].reshape(-1,1)]

    gradient = initial_gradient + regularisation
    return gradient.flatten()

def one_vs_all(inputs, outputs, number_of_labels, lambda_value): 
    number_of_samples, number_of_features = inputs.shape
    input_theta = np.zeros((number_of_features, 1))
    theta_values = np.zeros((number_of_labels, number_of_features))

    for classes in np.arange(1, number_of_labels+1): 
        optimised_cost_function = minimize(regularised_cost_function, 
                                           input_theta, 
                                           args=(lambda_value, inputs, (outputs == classes)*1),
                                           method=None,
                                           jac=regularised_gradient,
                                           options={'maxiter':50})
        theta_values[classes-1] = optimised_cost_function.x
    return theta_values

def compute_accuracy(theta_values, inputs, outputs):
    probability = sigmoid(inputs @ theta_values.T)
    probability = np.argmax(probability, axis=1) + 1
    return np.mean(probability == outputs.ravel()) * 100

def multiclass_logistic_regression(inputs, outputs):
    classifier = LogisticRegression(C=10, penalty='l2', solver='liblinear')
    classifier.fit(inputs[:,1:], outputs.ravel())

    prediction = classifier.predict(inputs[:, 1:])
    return np.mean(prediction == outputs.ravel()) * 100

def neural_networks(theta_1, theta_2, inputs, outputs): 
    number_of_samples = outputs.size

    output_2 = theta_1 @ inputs.T
    activation_2 = np.c_[np.ones((number_of_samples,1)), sigmoid(output_2).T]

    output_3 = activation_2 @ theta_2.T
    activation_3 = sigmoid(output_3)

    prediction = np.argmax(activation_3, axis=1) + 1
    return np.mean(prediction == outputs.ravel()) * 100


if __name__ == '__main__': 
    file_path = 'data/ex3data1.mat'
    digits_data = loadmat(file_path) 
    inputs, outputs = extract_variables(digits_data)

    visualise_sample(inputs) 

    n_labels = 10
    lambda_value = 0.1
    theta_values = one_vs_all(inputs, outputs, number_of_labels, lambda_value)
    accuracy = compute_accuracy(theta_values, inputs, outputs)

    multiclass_logistic_regression_accuracy = multiclass_logistic_regression(inputs, outputs)

    weights = loadmat('data/ex3weights.mat')
    theta_1, theta_2 = weights['Theta1'], weights['Theta2']
    neural_networks_accuracy = neural_networks(theta_1, theta_2, inputs, outputs)
