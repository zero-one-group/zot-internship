import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.optimize as opt 
from scipy.optimize import minimize 
from sklearn.preprocessing import PolynomialFeatures


def clean_data(file_path, variables): 
    data = pd.read_csv(file_path)
    data.columns = variables
    return data

def extract_variables(data): 
    inputs = data.iloc[:, :-1]
    number_of_samples = len(inputs) 
    intercept_variable = np.ones(number_of_samples) 
    inputs = np.c_[intercept_variable, inputs]

    outputs = data.iloc[:,2]
    outputs = outputs[:, np.newaxis]
    return inputs, outputs

def sigmoid(z): 
    return 1 / (1+np.exp(-z))

def compute_cost(theta, inputs, outputs):
    number_of_samples = len(inputs)
    predictions = sigmoid(inputs @ theta)

    negative_cost = np.log(predictions).T @ outputs
    positive_cost = np.log(1-predictions).T @ (1-outputs)
    cost = (-1/number_of_samples) * (negative_cost + positive_cost)
    return sum(cost)

def compute_gradient(theta, inputs, outputs):
    number_of_samples = len(outputs) 
    predictions = sigmoid(inputs @ theta.reshape(-1, 1))

    delta = inputs.T @ (predictions-outputs) 
    gradient = (1/number_of_samples) * delta 
    return gradient.flatten()

def extract_optimised_theta(theta, inputs, outputs):
    def compute_optimised_cost_function(theta, inputs, outputs): 
        return minimize(compute_cost, 
                        theta, 
                        args=(inputs, outputs), 
                        method=None, 
                        jac=compute_gradient, 
                        options={'maxiter':400})

    optimised_cost_function = compute_optimised_cost_function(theta, inputs, outputs)
    return optimised_cost_function.x

def generate_samples(samples):
    return np.array([samples])

def compute_probability(samples, theta):
    samples = generate_samples(samples)
    return sigmoid(np.array(samples).dot(optimised_theta.T))

def accuracy(theta, inputs, outputs, threshold): 
    probability = sigmoid(inputs @ theta.T) >= threshold
    probability = probability.astype('int') 
    return 100 * sum(probability == outputs.ravel())/probability.size

def plot_data(data, xlabel, ylabel): 
    def extract_inputs(data): 
        return data.iloc[:, 0]

    def extract_outputs(data): 
        return data.iloc[:, 1]

    admitted = data[data.iloc[:, -1] == 1]
    not_admitted = data[data.iloc[:, -1] == 0]
    plt.scatter(extract_inputs(admitted), extract_outputs(admitted), marker='+', color='black', label='passed')
    plt.scatter(extract_inputs(not_admitted), extract_outputs(not_admitted), marker='o', color='yellow', label='failed admitted', edgecolor='black') 
    plt.legend(loc='upper right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('figure 1: scatter plot of training data')

    return plt.show() 

def insert_polynomial_features(inputs): 
    inputs = inputs[:, 1:3]
    polynomial_features = PolynomialFeatures(6)
    return polynomial_features.fit_transform(inputs)

def compute_regularised_cost(theta, lambda_value, inputs, outputs):
    number_of_samples = outputs.size
    predictions = sigmoid(inputs @ theta)

    negative_cost = np.log(predictions).T @ (outputs)
    positive_cost = np.log(1-predictions).T @ (1-outputs)
    regularisation = (lambda_value/(2*number_of_samples)) * np.sum(np.square(theta[1:]))

    cost = (-1/number_of_samples) * (negative_cost + positive_cost) + regularisation 
    return sum(cost)

def compute_regularised_gradient(theta, lambda_value, x, y):
    number_of_samples = y.size
    predictions = sigmoid(inputs @ theta.reshape(-1,1))

    intial_gradient = (1/number_of_samples) * inputs.T @ (predictions-outputs) 
    regularisation = (lambda_value/number_of_samples) * np.r_[[[0]], theta[1:].reshape(-1,1)]
    gradient = initial_gradient + regularisation
    return gradient.flatten()


if __name__ == '__main__':
    # PART 1
    file_path = 'data/ex2data1.txt'
    column_names = ['score_1', 'score_2', 'admitted']
    exam_data = clean_data(file_path, column_names)

    xlabel = 'exam_score_1'
    ylabel = 'exam_score_2'
    plot_data(exam_data, xlabel, ylabel)

    inputs, outputs = extract_variables(exam_data)
    number_of_samples, number_of_features = inputs.shape
    theta = np.zeros(number_of_features)

    cost = compute_cost(theta, inputs, outputs)
    optimised_theta = extract_optimised_theta(theta, inputs, outputs)

    samples = [1, 45, 85]
    sample_admission_probability = compute_probability(samples, optimised_theta)

    threshold = 0.5
    accuracy = accuracy(optimised_theta, inputs, outputs, threshold)


    # PART 2 
    file_path = 'data/ex2data2.txt' 
    column_names = ['test_2', 'test_2', 'pass']
    microchip_data = clean_data(file_path, column_names)

    xlabel = 'microchip test 1'
    ylabel = 'microchip test 2'
    plot_data(microchip_data, xlabel, ylabel)

    inputs, outputs = extract_variables(microchip_data)
    inputs = insert_polynomial_features(inputs)
    number_of_samples, number_of_features = inputs.shape
    theta = np.zeros(number_of_features)
    lambda_value = 1

    regularised_cost = compute_regularised_cost(theta, lambda_value, inputs, outputs)
