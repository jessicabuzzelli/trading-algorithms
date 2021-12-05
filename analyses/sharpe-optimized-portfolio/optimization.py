import datetime as dt
import numpy as np
import pandas as pd
from analyses.utils import get_data, plot_experiment


def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), syms=["GOOG", "AAPL", "GLD", "XOM"], gen_plot=True):
    import scipy.optimize as optimize

    prices_all = get_data(syms, pd.date_range(sd, ed))
    prices = prices_all[syms]
    prices_SPY = prices_all["SPY"]

    # Initialize allocations
    portfolio_allocations = np.array([1 / len(syms) for sym in syms])

    # Find optimal allocations
    bounds = [(0.0, 1.0) for sym in syms]
    constraints = {'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)}

    normed = prices / prices.iloc[0, :]

    def neg_sharpe_ratio(portfolio_allocations):
        daily_port_vals = (normed * portfolio_allocations).sum(axis=1)
        daily_returns = ((daily_port_vals / daily_port_vals.shift(1)) - 1)[1:]
        return np.sqrt(252) * -daily_returns.mean() / daily_returns.std()

    opt_sharpe_solver = optimize.minimize(fun=neg_sharpe_ratio,
                                          x0=portfolio_allocations,
                                          method='SLSQP',
                                          constraints=constraints,
                                          bounds=bounds)

    if opt_sharpe_solver.success:
        portfolio_allocations = opt_sharpe_solver.x
        sr = -opt_sharpe_solver.fun

        daily_port_vals = (normed * portfolio_allocations).sum(axis=1)
        daily_port_returns = ((daily_port_vals / daily_port_vals.shift(1)) - 1)[1:]

        daily_SPY_vals = prices_SPY / prices_SPY.iloc[0]

        sddr = daily_port_returns.std()
        adr = daily_port_returns.mean()
        cr = (daily_port_vals[-1] - daily_port_vals[0]) / daily_port_vals[0]

        if gen_plot:
            df = pd.concat(
                [daily_port_vals, daily_SPY_vals],
                keys=["Portfolio", "SPY"],
                axis=1
            )

            plot_experiment(df,
                            title='Daily Portfolio Value and SPY',
                            xlabel='Date',
                            ylabel='Price',
                            legend=["Portfolio", "SPY"],
                            fname='optimized_portfolio')

        return portfolio_allocations, cr, adr, sddr, sr

    else:
        print(opt_sharpe_solver.message)
        return


def test(start_date=dt.datetime(2008, 6, 1), end_date=dt.datetime(2009, 6, 1), symbols=["IBM", "X", "GLD", "JPM"], verbose=True):
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols
    )

    assert sum(allocations) == 1

    if verbose:
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Symbols: {symbols}")
        print(f"Allocations:{allocations}")
        print(f"Sharpe Ratio: {sr}")
        print(f"Volatility (stdev of daily returns): {sddr}")
        print(f"Average Daily Return: {adr}")
        print(f"Cumulative Return: {cr}")


if __name__ == '__main__':
    test()
