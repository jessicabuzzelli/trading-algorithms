import numpy as np


class LinearRegressionLearner:
    def __init__(self):
        self.coefficients = None

    def add_evidence(self, x_train, y_train):
        self.coefficients = np.linalg.lstsq(x_train, y_train)[0]

    def query(self, x_test):
        return (self.coefficients[:-1] * x_test).sum(axis=1) + self.coefficients[-1]
