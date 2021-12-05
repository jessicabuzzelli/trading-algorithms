import numpy as np
import random


class ForestLearner:
    def __init__(self, learner, bags=1, kwargs={}):
        self.learners = []
        for x in range(bags):
            self.learners.append(learner(**kwargs))

        self.bags = bags
        self.kwargs = kwargs

    def add_evidence(self, x_train, y_train):
        n, m = x_train.shape

        for learner in self.learners:
            train_idxs = random.choices([x for x in range(n)], k=n)
            learner.add_evidence(x_train[train_idxs, :], y_train[train_idxs])

    def query(self, x_test):
        y_pred = np.array([learner.query(x_test) for learner in self.learners])
        return np.mean(y_pred, axis=0)
