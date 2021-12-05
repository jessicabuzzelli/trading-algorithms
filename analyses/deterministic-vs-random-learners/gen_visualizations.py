import numpy as np
import pandas as pd
import sys
import random
from timeit import default_timer as timer
from learners.DecisionTreeLearner import DecisionTreeLearner
from learners.RandomTreeLearner import RandomTreeLearner
from learners.ForestLearner import ForestLearner
from analyses.utils import plot_experiment


def data_setup(f):
    data = np.genfromtxt(f, delimiter=',')

    if all(np.isnan(data[0, :])):
        data = data[1:, :]

    if all(np.isnan(data[:, 0])):
        data[:, 0] = np.array([x for x in range(data.shape[0])])

    train_rows = int(0.6 * data.shape[0])

    random.shuffle(data)
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    return train_x, train_y, test_x, test_y


def rmse(test_y, y_pred):
    return np.sqrt(((test_y - y_pred) ** 2).sum() / test_y.shape[0])


def mae(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred)))


def experiment_one(train_x, train_y, test_x, test_y):
    test_leaf_sizes = [x for x in range(1, int(train_x.shape[0] * .6), 1)]
    learners = []

    out_sample_rmse_by_leaf_size = {}
    in_sample_rmse_by_leaf_size = {}

    for size in test_leaf_sizes:
        learners.append(DecisionTreeLearner(leaf_size=size))

    for learner in learners:
        learner.add_evidence(train_x, train_y)

        y_pred_train = learner.query(train_x)
        y_pred_test = learner.query(test_x)

        in_sample_rmse_by_leaf_size[learner.leaf_size] = rmse(train_y, y_pred_train)
        out_sample_rmse_by_leaf_size[learner.leaf_size] = rmse(test_y, y_pred_test)

    df = pd.DataFrame.from_dict(in_sample_rmse_by_leaf_size, orient='index', columns=['In-Sample RMSE'])
    df2 = pd.DataFrame.from_dict(out_sample_rmse_by_leaf_size, orient='index', columns=['Out-Sample RMSE'])
    df['Out-Sample RMSE'] = df2['Out-Sample RMSE']

    plot_experiment(df,
                    title='Decision Tree Learners: RMSE vs. Leaf Size',
                    xlabel='Leaf Size',
                    ylabel='RMSE',
                    legend=['In-Sample', 'Out-Sample'],
                    fname='experiment_one')


def experiment_two(train_x, train_y, test_x, test_y, bags=5):
    test_leaf_sizes = [x for x in range(1, int(train_x.shape[0] * .6), 1)]
    learners = []

    out_sample_rmse_by_leaf_size = {}
    in_sample_rmse_by_leaf_size = {}

    for size in test_leaf_sizes:
        learners.append(ForestLearner(bags=bags, learner=DecisionTreeLearner, kwargs={'leaf_size': size}))

    for learner in learners:
        learner.add_evidence(train_x, train_y)

        y_pred_train = learner.query(train_x)
        y_pred_test = learner.query(test_x)

        in_sample_rmse_by_leaf_size[learner.kwargs['leaf_size']] = rmse(train_y, y_pred_train)
        out_sample_rmse_by_leaf_size[learner.kwargs['leaf_size']] = rmse(test_y, y_pred_test)

    df = pd.DataFrame.from_dict(in_sample_rmse_by_leaf_size, orient='index', columns=['In Sample RMSE'])
    df2 = pd.DataFrame.from_dict(out_sample_rmse_by_leaf_size, orient='index', columns=['Out-Sample RMSE'])
    df['Out-Sample RMSE'] = df2['Out-Sample RMSE']

    plot_experiment(df,
                    title='Bagged Decision Tree Learners (n={bags}): RMSE vs. Leaf Size',
                    xlabel='Leaf Size',
                    ylabel='RMSE',
                    legend=['In-Sample', 'Out-Sample'],
                    fname=f'experiment_two_{bags}_bags')


def experiment_three(train_x, train_y, test_x, test_y):
    test_leaf_sizes = [x for x in range(1, int(train_x.shape[0] * .6), 1)]
    dt_learners = []
    rt_learners = []

    dt_mae_by_leaf_size = {}
    rt_mae_by_leaf_size = {}

    dt_train_time_by_leaf_size = {}
    rt_train_time_by_leaf_size = {}

    for size in test_leaf_sizes:
        dt_learners.append(DecisionTreeLearner(leaf_size=size))
        rt_learners.append(RandomTreeLearner(leaf_size=size))

    for learner in dt_learners:
        start = timer()
        learner.add_evidence(train_x, train_y)
        end = timer()

        dt_train_time_by_leaf_size[learner.leaf_size] = (end - start) * 1000
        dt_mae_by_leaf_size[learner.leaf_size] = mae(test_y, learner.query(test_x))

    for learner in rt_learners:
        start = timer()
        learner.add_evidence(train_x, train_y)
        end = timer()

        rt_train_time_by_leaf_size[learner.leaf_size] = (end - start) * 1000
        rt_mae_by_leaf_size[learner.leaf_size] = mae(test_y, learner.query(test_x))

    time_to_train_df = pd.DataFrame.from_dict(rt_train_time_by_leaf_size, orient='index', columns=['RandomTreeLearner'])
    time_to_train_df['DecisionTreeLearner'] = pd.DataFrame.from_dict(dt_train_time_by_leaf_size, orient='index', columns=['DecisionTreeLearner'])['DecisionTreeLearner']

    mae_df = pd.DataFrame.from_dict(rt_mae_by_leaf_size, orient='index', columns=['RandomTreeLearner'])
    mae_df['DecisionTreeLearner'] = pd.DataFrame.from_dict(dt_mae_by_leaf_size, orient='index', columns=['DecisionTreeLearner'])['DecisionTreeLearner']

    plot_experiment(time_to_train_df,
                    title='Decision Tree vs. Random Tree: Time to Train by Leaf Size',
                    xlabel='Leaf Size',
                    ylabel='Time to Train (ms)',
                    legend=['RandomTreeLearner', 'DecisionTreeLearner'],
                    fname='experiment_three_time_to_train')

    plot_experiment(mae_df,
                    title='Decision Tree vs. Random Tree: Mean Absolute Error (MAE) by Leaf Size',
                    xlabel='Leaf Size',
                    ylabel='MAE',
                    legend=['RandomTreeLearner', 'DecisionTreeLearner'],
                    fname='experiment_three_mae')


def main(f):
    try:
        train_x, train_y, test_x, test_y = data_setup(f)

    except FileNotFoundError:
        print('Data file not found.')
        return

    except OSError:
        print('Data file not found.')
        return

    experiment_one(train_x, train_y, test_x, test_y)

    experiment_two(train_x, train_y, test_x, test_y, bags=5)

    experiment_two(train_x, train_y, test_x, test_y, bags=10)

    experiment_three(train_x, train_y, test_x, test_y)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python gen_visualizations.py <data filename>')

    else:
        random.seed(1)
        main(f=sys.argv[1])
