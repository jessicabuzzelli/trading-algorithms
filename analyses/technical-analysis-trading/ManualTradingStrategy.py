import pandas as pd
from analyses.utils import getData, calculateIndicators


class ManualTradingStrategy:
    def __init__(self):
        pass

    def testPolicy(self, symbol, sd, ed, period=10):
        prices = getData([symbol], pd.date_range(sd, ed))[[symbol]].sort_index(ascending=True)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        prices = prices / prices.iloc[0]

        psma, bbp, momentum = calculateIndicators(prices, window=period)
        x_train = pd.concat((psma, bbp, momentum), axis=1).fillna(0)[:-period]
        x_train.columns = ['PSMA', 'BBP', 'Momentum']

        trades = pd.DataFrame(index=prices.index, columns=['Shares'])
        shares = 0
        last_price = 0

        for i in range(prices.shape[0] - period):
            date = prices.index[i]
            price = prices.loc[date, 'Price']
            if shares == 1000:
                if x_train.loc[date, 'PSMA'] < -0.05 or x_train.loc[date, 'BBP'] < -0.8 or x_train.loc[date, 'Momentum'] < -0.03:
                    if last_price < price:
                        trades.iloc[i, 0] = -2000
                        shares = -1000
                        last_price = price
            elif shares == -1000:
                if x_train.loc[date, 'PSMA'] > 0.05 or x_train.loc[date, 'BBP'] > 0.8 or x_train.loc[date, 'Momentum'] > -0.03:
                    if last_price > price:
                        trades.iloc[i, 0] = 2000
                        shares = 1000
                        last_price = price
            else:
                if x_train.loc[date, 'PSMA'] < -0.05 or x_train.loc[date, 'BBP'] < -0.8 or x_train.loc[date, 'Momentum'] < -0.03:
                    if last_price < price:
                        trades.iloc[i, 0] = -1000
                        shares = -1000
                        last_price = price
                elif x_train.loc[date, 'PSMA'] > 0.05 or x_train.loc[date, 'BBP'] > 0.8 or x_train.loc[date, 'Momentum'] > -0.03:
                    if last_price > price:
                        trades.iloc[i, 0] = 1000
                        shares = 1000
                        last_price = price
                else:
                    pass

        if shares == 1000:
            trades.iloc[-1, 0] = -1000

        if shares == -1000:
            trades.iloc[-1, 0] = 1000

        trades.fillna(0, inplace=True)
        return trades[trades['Shares'] != 0]
