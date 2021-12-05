import numpy as np


class RandomTreeLearner(object):
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.tree = None

    def build_tree(self, train_x, train_y):
        n, m = train_x.shape

        if n <= self.leaf_size:
            return np.array([-1, train_y.mean(), np.nan, np.nan])

        if np.unique(train_y).shape[0] == 1:
            return np.array([-1, train_y.mean(), np.nan, np.nan])

        if np.unique(train_x).shape[0] == 1:
            return np.array([-1, train_y.mean(), np.nan, np.nan])

        splittable_features = np.array([x for x in range(m)])
        split_feature = np.random.choice(splittable_features)

        # remove split feature from list of possible features if uniform across datapoints
        while np.unique(train_x[:, split_feature]).shape[0] == 1 and len(splittable_features) > 1:
            splittable_features = splittable_features[splittable_features != split_feature]
            split_feature = np.random.choice(splittable_features)

        # can't split if each feature has same x value(s) across all observations
        if len(splittable_features) == 1 and np.unique(train_x[:, splittable_features[0]]).shape[0] == 1:
            return np.array([-1, train_y.mean(), np.nan, np.nan])

        # can't split if each feature has same x value(s) across all observations
        elif len(splittable_features) == 0:
            return np.array([-1, train_y.mean(), np.nan, np.nan])

        else:
            split_val = np.median(train_x[:, split_feature])
            left_index = train_x[:, split_feature] <= split_val
            right_index = ~left_index

            if all(left_index) or all(right_index):
                return np.array([-1, train_y.mean(), np.nan, np.nan])

            lefttree = self.build_tree(train_x[left_index], train_y[left_index])
            righttree = self.build_tree(train_x[right_index], train_y[right_index])

            if lefttree.ndim is 1:
                righttree_start = 2

            else:
                righttree_start = lefttree.shape[0] + 1

            root = np.array([split_feature, split_val, 1, righttree_start])

            return np.vstack((root, lefttree, righttree))

    def add_evidence(self, train_x, train_y):
        self.tree = None
        self.train_x = train_x
        self.train_y = train_y

        if train_x.shape[0] <= self.leaf_size:
            self.tree = np.array([-1, train_y.mean(), np.nan, np.nan])

        else:
            self.tree = self.build_tree(train_x, train_y)

    def tree_lookup(self, vec, node=0):
        split_feature = self.tree[node, 0]
        split_val = self.tree[node, 1]

        # found leaf, quit recursion
        if split_feature == -1:
            return split_val

        # go left
        elif vec[int(split_feature)] <= split_val:
            return self.tree_lookup(vec, node + int(self.tree[node, 2]))

        # go right
        else:
            return self.tree_lookup(vec, node + int(self.tree[node, 3]))

    def query(self, test_x):
        if len(self.tree) == 4:
            return np.array([1] + [0 for vec in test_x[1:]])

        else:
            return np.array([self.tree_lookup(vec) for vec in test_x])
