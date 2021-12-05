import numpy as np
from scipy.stats import pearsonr


class DecisionTreeLearner(object):
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.tree = None

    def build_tree(self, x_train, y_train):
        n, m = x_train.shape

        if n <= self.leaf_size or np.unique(y_train).shape[0] == 1:
            return np.array([-1, y_train[0], np.nan, np.nan])

        corr_dict = {feature: abs(pearsonr(x_train[:, feature], y_train)[0]) for feature in range(m)}
        split_feature = max(corr_dict, key=corr_dict.get)

        while np.unique(x_train[:, split_feature]).shape[0] == 1 and corr_dict:
            del corr_dict[split_feature]
            split_feature = max(corr_dict, key=corr_dict.get)

        # can't split if each feature has same x value(s) across all observations
        if not corr_dict:
            return np.array([-1, y_train.mean(), np.nan, np.nan])

        else:
            split_val = np.median(x_train[:, split_feature])

            left_index = x_train[:, split_feature] <= split_val
            right_index = ~left_index

            if all(left_index) or all(right_index):
                return np.array([-1, y_train.mean(), np.nan, np.nan])

            lefttree = self.build_tree(x_train[left_index], y_train[left_index])
            righttree = self.build_tree(x_train[right_index], y_train[right_index])

            if lefttree.ndim == 1:
                root = np.array([split_feature, split_val, 1, 2])
            else:
                root = np.array([split_feature, split_val, 1, lefttree.shape[0] + 1])

            return np.vstack((root, lefttree, righttree))

    def tree_lookup(self, vec, node=0):
        split_feature = self.tree[node, 0]
        split_val = self.tree[node, 1]

        # found leaf, quit recursion
        if split_feature == -1:
            return split_val

        # go left
        elif vec[int(split_feature)] <= split_val:
            pred = self.tree_lookup(vec, node + int(self.tree[node, 2]))

        # go right
        else:
            pred = self.tree_lookup(vec, node + int(self.tree[node, 3]))

        return pred

    def add_evidence(self, x_train, y_train):
        self.tree = None
        self.tree = self.build_tree(x_train, y_train)

    def query(self, x_test):
        y_pred = []
        for vec in x_test:
            y_pred.append(self.tree_lookup(vec))

        return np.asarray(y_pred)
