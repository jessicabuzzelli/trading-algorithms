import matplotlib.pyplot as plt
import pandas as pd


def plotExperiment(df, title, xlabel, ylabel, legend, fname):
    plt.plot(df)
    plt.grid(True, linestyle='--',)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    plt.savefig(f'{fname}.png')
    plt.clf()


def getData(symbols, dates, addSPY=True, colname="Adj Close"):
    pass


def getStats(portvals):
    pv = portvals
    dr = np.diff(pv) / pv[1:]

    mdr = dr.mean()

    stddr = dr.std()

    cr = (pv[-1] / pv[0]) - 1

    return cr, mdr, stddr


def computePortvals(symbol, trades_df, sv, sd, ed, impact, commission):
    orders_df = pd.DataFrame(columns=["Shares", "Order", "Symbol"])

    orders_df['Shares'] = trades_df['Shares']
    orders_df = orders_df[orders_df['Shares'] != 0]
    orders_df.loc[orders_df['Shares'] > 0, 'Order'] = 'BUY'
    orders_df.loc[orders_df['Shares'] < 0, 'Order'] = 'SELL'

    orders_df.loc[:, 'Shares'] = abs(orders_df['Shares'])

    orders_df['Symbol'] = symbol

    return runMarketSimulator(orders_df, sd, ed, sv, impact, commission)


def runMarketSimulator(orders_df, start_date, end_date, sv, impact=0.0, commission=0.0):
    symbols = orders_df['Symbol'].unique()

    orders_df = orders_df.sort_index()

    prices = getData(symbols, pd.date_range(start_date, end_date))[symbols].fillna(method="ffill").fillna(method="bfill")
    prices["CASH"] = 1.0

    trades_df = pd.DataFrame(index=prices.index, columns=symbols).fillna(0)

    cash = pd.Series(index=prices.index).fillna(0)
    cash.iloc[0] = sv

    for date, order in orders_df.iterrows():
        shares = order["Shares"]
        action = order["Order"]
        symbol = order["Symbol"]
        price = prices.loc[date, symbol]

        if action == "SELL":
            shares *= -1

        cost = price * shares
        cost += commission + (price * abs(shares) * impact)

        trades_df.loc[date, symbol] += shares
        cash.loc[date] -= cost

    trades_df["CASH"] = cash
    portvals_df = (prices * trades_df.cumsum()).sum(axis=1)

    return portvals_df


def calculateIndicators(df, window=10):
    df.columns = ["Price"]

    sma = df["Price"].rolling(window).mean()
    rstd = df["Price"].rolling(window).std()

    psma = df['Price'] / sma

    upper_bollinger_band = sma + (rstd * 2)
    lower_bollinger_band = sma - (rstd * 2)
    bbp = (df["Price"] - lower_bollinger_band) / (upper_bollinger_band - lower_bollinger_band)

    momentum = (df["Price"] / df["Price"].shift(window)) - 1

    return psma, bbp, momentum
