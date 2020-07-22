import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.columns = ['population', 'profit']
    return data

def extract_features(data):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    return x, y

def add_axis(x, y):
    x = x[:, np.newaxis]

    number_of_samples = len(x)
    ones = np.ones((number_of_samples,1))
    x = np.hstack((ones, x))
    y = y[:, np.newaxis]
    return x, y

def predict(x, theta): 
    return x @ theta

def gradient_descent(x, y, theta, alpha, iterations): 
    x, y = add_axis(x, y)
    number_of_samples = len(y)

    for iteration in range(iterations): 
        hypothesis = predict(x, theta)
        difference = hypothesis - y 
        slope = np.dot(x.T, difference) 
        theta = theta - (alpha/number_of_samples) * slope
    return theta

def cost(x, y, theta): 
    x, y = add_axis(x, y)
    number_of_samples = len(y) 

    hypothesis = predict(x, theta)
    difference = hypothesis - y
    square_difference = np.power(difference, 2)
    sum_of_square_difference = np.sum(square_difference)
    return sum_of_square_difference/(2*m)

def plot_cost(x, y, theta): 
    plt.scatter(x, y, marker='x', color='red')
    plt.xlim([3, 25])
    plt.ylim([-5, 25])
    plt.xlabel('Population of City in 10,000s', fontsize=10)
    plt.ylabel('Profit in $10,000s', fontsize=10)
    plt.title('Figure 1: Scatter plot of training data', fontsize=15)

    x, y = add_axis(x, y)
    hypothesis = predict(x, theta)

    plt.plot(x, hypothesis, color='blue')
    return plt.show()

def compute_hypothesis(theta, population): 
    intercept = theta[0]
    slope = theta[1]
    return intercept + slope * population


if __name__ == '__main__':
    file_path = 'data/ex1data1.txt'
    profit_data = clean_data(file_path)

    x, y = extract_features(profit_data)

    theta = np.zeros([2,1])
    iterations = 1500
    alpha = 0.01

    theta = gradient_descent(x, y, theta, alpha, iterations)
    cost = cost(x, y, theta)
    print('theta = {}, cost = {}'.format(theta, cost))

    plot_cost(x, y, theta)

    population = 7
    population_hypothesis = compute_hypothesis(theta, population)
    print('population hypothesis = {}'.format(population_hypothesis))
