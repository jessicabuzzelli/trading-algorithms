import pandas as pd
import numpy as np
from learners.ForestLearner import ForestLearner
from learners.RandomTreeLearner import RandomTreeLearner
from analyses.utils import getData, calculateIndicators


class TechnicalIndicatorLearner:

    def __init__(self, verbose=False, impact=0.0, commission=0.0, period=5, bags=5, leaf_size=5):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.period = period
        self.learner = ForestLearner(learner=RandomTreeLearner, bags=bags, kwargs={"leaf_size": leaf_size})

    def add_evidence(self, symbol, sd, ed, y_buy=0.02, y_sell=0.02):
        prices = getData([symbol], pd.date_range(sd, ed))[[symbol]].sort_index(ascending=True)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        prices = prices / prices.iloc[0]

        psma, bbp, momentum = calculateIndicators(prices, window=self.period)
        x_train = pd.concat((psma, bbp, momentum), axis=1).fillna(0)[:-self.period]

        daily_rets = ((prices.values[self.period:] / prices.values[:-self.period]) - 1).T[0]
        buy = y_buy + self.impact
        sell = -(y_sell + self.impact)

        y_train = np.array((daily_rets > buy).astype(int) - (daily_rets < sell).astype(int))

        if self.verbose:
            print(f"Number of trade signals in training set: {y_train[y_train != 0].shape}")

        self.learner.add_evidence(x_train.values, y_train)

    def testPolicy(self, symbol, sd, ed, sv):
        prices = getData([symbol], pd.date_range(sd, ed))[[symbol]].sort_index(ascending=True)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        prices = prices / prices.iloc[0]

        psma, bbp, momentum = calculateIndicators(prices, window=self.period)
        x_test = pd.concat((psma, bbp, momentum), axis=1).fillna(0).values

        y_test = self.learner.query(x_test)

        if self.verbose:
            print(f"# of trades recommended: {y_test[y_test != 0].shape}")

        trades = pd.DataFrame(index=prices.index, columns=['Shares'])
        shares = 0
        for i in range(y_test.shape[0] - 1):
            if y_test[i] > 0:
                if shares == 0:
                    trades.iloc[i, 0] = 1000
                    shares = 1000
                if shares == -1000:
                    trades.iloc[i, 0] = 2000
                    shares = 1000
            if y_test[i] < 0:
                if shares == 0:
                    trades.iloc[i, 0] = -1000
                    shares = -1000
                if shares == 1000:
                    trades.iloc[i, 0] = -2000
                    shares = -1000

        if shares == -1000:
            trades.iloc[-1, 0] = 1000
        if shares == 1000:
            trades.iloc[-1, 0] = -1000

        trades.fillna(0, inplace=True)
        trades = trades[trades['Shares'] != 0]

        if self.verbose:
            print(f"Number of trades: {trades.shape}")

        return trades
