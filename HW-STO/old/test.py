import numpy as np
import os
# w = np.loadtxt("filename.txt", delimiter=',')
from keras import datasets ## 载入数据

mnist = datasets.mnist
train_data, test_data = mnist.load_data()

def test(filename, test_data):
    w = np.loadtxt(filename, delimiter=',')
    x , y = test_data
    x = x / 255.
    y = y % 2
    wx = np.zeros(len(y))
    for _ in range(len(y)):
        x_ = x[_].flatten()
        wx[_] = np.dot(w,x_)
    wx[wx>0.5] = 1
    wx[wx<=0.5] = 0
    accuracy = np.sum(1*(wx == y)) / len(y)
    print('accuracy is '+str(accuracy) )
    return accuracy


test('./adam_results/adam_result_with_lambda_equals_to_0.001', train_data)
test('./adam_results/adam_result_with_lambda_equals_to_0.1', test_data)
test('./adam_results/adam_result_with_lambda_equals_to_1', test_data)
test('./adam_results/adam_result_with_lambda_equals_to_10', test_data)

test('./Nesterov_results/Nesterov_result_with_lambda_equals_to_0.0001', test_data)
test('./Nesterov_results/Nesterov_result_with_lambda_equals_to_0.1', test_data)
test('./Nesterov_results/Nesterov_result_with_lambda_equals_to_1', test_data)
test('./Nesterov_results/Nesterov_result_with_lambda_equals_to_10', test_data)