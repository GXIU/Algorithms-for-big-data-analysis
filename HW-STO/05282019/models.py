'''algorithms for this problem'''
import numpy as np
import tensorflow as tf
import logging as log

class Classifier(object):
    def __init__(self, i_stepwize = 0.01):
        self.i_stepwize = i_stepwize
        self.size = None
        self.w = None
    def prediction(self, w, X):
        if X.ndim == 2:
            size = X.shape[0]
        p = np.zeros(size)

        for _ in range(size):
            wTX = np.dot(X[i], w)
            prob = 1./(1.+np.exp)
    