import pandas as pd
import datetime as dt
import random
import matplotlib.pyplot as plt
from analyses.utils import getData, computePortvals, getStats
from ManualTradingStrategy import ManualTradingStrategy
from TechnicalIndicatorLearner import TechnicalIndicatorLearner


def build_benchmark_orders(idx):
    benchmark_trades = pd.DataFrame(columns=['Shares'], index=idx).fillna(0)
    benchmark_trades.iloc[0] = [1000]
    benchmark_trades.iloc[-1] = [-1000]

    return benchmark_trades[benchmark_trades['Shares'] != 0]


def manual_strategy_vs_benchmark(in_sample=True, symbol='JPM', impact=0.005, commission=9.95, sv=100000, verbose=False):
    if in_sample:
        sd, ed = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
        fname = 'In Sample'

    else:
        sd, ed = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)
        fname = 'Out Sample'

    prices = getData([symbol], pd.date_range(sd, ed))

    benchmark_orders_df = build_benchmark_orders(prices.index)
    benchmark_portvals = computePortvals(symbol, benchmark_orders_df, sv, sd, ed, impact, commission)

    manual_strategy_trades = ManualTradingStrategy().testPolicy(symbol=symbol, sd=sd, ed=ed)
    manual_strategy_portvals = computePortvals(symbol, manual_strategy_trades, sv, sd, ed, impact, commission)

    if verbose:
        cr, mdr, stddr = getStats(benchmark_portvals)
        print("Benchmark:")
        print(f"    Cumulative Return: {cr}")
        print(f"    Mean Daily Return: {mdr}")
        print(f"    Standard Deviation of Daily Returns: {stddr}")
        print()

        cr, mdr, stddr = getStats(manual_strategy_portvals)
        print("Manual Strategy:")
        print(f"    Cumulative Return: {cr}")
        print(f"    Mean Daily Return: {mdr}")
        print(f"    Standard Deviation of Daily Returns: {stddr}")
        print()

    (benchmark_portvals/benchmark_portvals[0]).plot(color="green", label='Benchmark')
    (manual_strategy_portvals/manual_strategy_portvals[0]).plot(color="red", label="Manual Strategy")

    long_dates = manual_strategy_trades[manual_strategy_trades['Shares'] > 0].index.values
    short_dates = manual_strategy_trades[manual_strategy_trades['Shares'] < 0].index.values

    for d in short_dates:
        plt.axvline(x=d, color='black')

    for d in long_dates:
        plt.axvline(x=d, color='blue')

    plt.grid(True, linestyle='--')
    plt.title(f"{fname}: Manual Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig(f'manual_strategy_{fname}.png')

    plt.clf()


def strategy_learner_vs_benchmark(in_sample=True, symbol='JPM', impact=0.005, commission=9.95, sv=100000, period=10,
                                  bags=15, leaf_size=5, verbose=False):
    sd, ed = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)

    if in_sample:
        sd1, ed1 = dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)
        fname = 'In Sample'

    else:
        sd1, ed1 = dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)
        fname = 'Out Sample'

    prices = getData([symbol], pd.date_range(sd1, ed1))

    benchmark_orders_df = build_benchmark_orders(prices.index)
    benchmark_portvals = computePortvals(symbol, benchmark_orders_df, sv, sd1, ed1, impact, commission)

    sl = TechnicalIndicatorLearner(verbose=False, impact=impact, commission=commission, period=period, bags=bags, leaf_size=leaf_size)
    sl.add_evidence(symbol=symbol, sd=sd, ed=ed, y_buy=0.02, y_sell=0.02)
    strategy_learner_trades = sl.testPolicy(symbol=symbol, sd=sd1, ed=ed1, sv=sv)
    strategy_learner_portvals = computePortvals(symbol, strategy_learner_trades, sv, sd1, ed1, impact, commission)

    if verbose:
        cr, mdr, stddr = getStats(benchmark_portvals)
        print("Benchmark:")
        print(f"    Cumulative Return: {cr}")
        print(f"    Mean Daily Return: {mdr}")
        print(f"    Standard Deviation of Daily Returns: {stddr}")
        print()

        cr, mdr, stddr = getStats(strategy_learner_portvals)
        print("Strategy Learner:")
        print(f"    Cumulative Return: {cr}")
        print(f"    Mean Daily Return: {mdr}")
        print(f"    Standard Deviation of Daily Returns: {stddr}")
        print()

    (benchmark_portvals/benchmark_portvals[0]).plot(color="green", label='Benchmark')
    (strategy_learner_portvals/strategy_learner_portvals[0]).plot(color="red", label="Strategy Learner")

    long_dates = strategy_learner_trades[strategy_learner_trades['Shares'] > 0].index.values
    short_dates = strategy_learner_trades[strategy_learner_trades['Shares'] < 0].index.values

    for d in short_dates:
        plt.axvline(x=d, color='black')

    for d in long_dates:
        plt.axvline(x=d, color='blue')

    plt.grid(True, linestyle='--')
    plt.title(f"{fname}: Strategy Learner vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig(f'strategy_learner_{fname}.png')

    plt.clf()


def experiment1(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, impact=0.005, commission=9.95,
    period=10, bags=15, leaf_size=5, verbose=False):
    prices = getData([symbol], pd.date_range(sd, ed))

    benchmark_orders_df = build_benchmark_orders(prices.index)
    benchmark_portvals = computePortvals(symbol, benchmark_orders_df, sv, sd, ed, impact, commission)

    if verbose:
        cr, mdr, stddr = getStats(benchmark_portvals)
        print("Benchmark")
        print(f"Cumulative Return: {cr}")
        print(f"Mean Daily Return: {mdr}")
        print(f"Standard Deviation of Daily Returns: {stddr}")
        print()

    manual_strategy_trades = ManualTradingStrategy().testPolicy(symbol=symbol, sd=sd, ed=ed)
    manual_strategy_portvals = computePortvals(symbol, manual_strategy_trades, sv, sd, ed, impact, commission)

    if verbose:
        cr, mdr, stddr = getStats(manual_strategy_portvals)
        print("Manual Strategy")
        print(f"Cumulative Return: {cr}")
        print(f"Mean Daily Return: {mdr}")
        print(f"Standard Deviation of Daily Returns: {stddr}")
        print()

    sl = TechnicalIndicatorLearner(verbose=False, impact=impact, commission=commission, period=period, bags=bags, leaf_size=leaf_size)
    sl.add_evidence(symbol=symbol, sd=sd, ed=ed)
    strategy_learner_trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    strategy_learner_portvals = computePortvals(symbol, strategy_learner_trades, sv, sd, ed, impact, commission)

    if verbose:
        cr, mdr, stddr = getStats(strategy_learner_portvals)
        print("Strategy Learner")
        print(f"Cumulative Return: {cr}")
        print(f"Mean Daily Return: {mdr}")
        print(f"Standard Deviation of Daily Returns: {stddr}")
        print()

    (benchmark_portvals/benchmark_portvals[0]).plot(color="black", label='Benchmark')
    (manual_strategy_portvals/manual_strategy_portvals[0]).plot(color="red", label="Manual Strategy")
    (strategy_learner_portvals/strategy_learner_portvals[0]).plot(color="blue", label='Strategy Learner')

    plt.title("Experiment 1: Strategy Learner, Manual Strategy vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig('experiment1.png')
    plt.clf()


def experiment2(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, commission=0,
         period=10, bags=15, leaf_size=5, verbose=False):

    for impact, color in zip((0.005, 0.01, 0.02, 0.05), ('black', 'blue', 'red', 'green')):
        sl = TechnicalIndicatorLearner(verbose=verbose, impact=impact, commission=commission, period=period, bags=bags, leaf_size=leaf_size)
        sl.add_evidence(symbol=symbol, sd=sd, ed=ed)

        strategy_learner_trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
        strategy_learner_portvals = computePortvals(symbol, strategy_learner_trades, sv, sd, ed, impact, commission)

        (strategy_learner_portvals/strategy_learner_portvals[0]).plot(color=color, label=f"Impact= {impact}")

        if verbose:
            cr, mdr, stddr = getStats(strategy_learner_portvals)
            print(f"Strategy Learner, Impact= {impact}")
            print(f"Cumulative Return: {cr}")
            print(f"Mean Daily Return: {mdr}")
            print(f"Standard Deviation of Daily Returns: {stddr}")
            print()

    plt.title("Experiment 2: In-Sample Strategy Learner Portvals vs. Impact")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.savefig('experiment2.png')
    plt.clf()


def main(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, impact=0.005,
         commission=9.95, period=10, bags=5, leaf_size=5, verbose=False):
    experiment1(symbol=symbol, sd=sd, ed=ed, sv=sv, impact=impact, commission=commission, period=period, bags=bags,
                     leaf_size=leaf_size, verbose=verbose)
    experiment2(symbol=symbol, sd=sd, ed=ed, sv=sv, commission=0, period=period, bags=bags,
                     leaf_size=leaf_size, verbose=verbose)
    manual_strategy_vs_benchmark(in_sample=True, symbol=symbol, sv=sv, impact=impact, commission=commission,
                                 verbose=verbose)
    manual_strategy_vs_benchmark(in_sample=False, symbol=symbol, sv=sv, impact=impact, commission=commission,
                                 verbose=verbose)
    strategy_learner_vs_benchmark(in_sample=True, symbol=symbol, impact=impact, commission=commission, sv=sv,
                                  period=period, bags=bags, leaf_size=leaf_size, verbose=verbose)
    strategy_learner_vs_benchmark(in_sample=False, symbol=symbol, impact=impact, commission=commission, sv=sv,
                                  period=period, bags=bags, leaf_size=leaf_size, verbose=verbose)


if __name__ == '__main__':
    random.seed(1)
    main(verbose=False)
