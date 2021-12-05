### Technical Analysis Trading ###

Uses % Simple Moving Average, Bollinger Band Percentage, and price momentum to determine long/short signals for one stock over a time period.

### Structure ###
`testproject.py`
    - Runs Experiments 1 & 2, generates in-sample and out-sample charts for Strategy Learner and Manual Strategy vs a 1000 share benchmark

`indicators.py`
    - `calculate_indicators` returns values of 3 technical indicators (% of Simple Moving Avg., Bollinger Band %, and momentum)

`ManualStrategy.py`
    - `testPolicy`:
        - loads a training matrix of technical indicators using `indicators.calculate_indicators`
        - compares indicators to preset thresholds to determine a trading action (LONG or SHORT net 1000 shares)
        - returns `trades_df` containing the number of shares (negative to indicate SHORT positions) to be traded, indexed by date

`StrategyLearner.py`
    - `add_evidence`:
        - trains a Random Forest (`BagLearner.py`, `RTLearner.py`) using the training matrix where the output is -1, 0, or 1 to indicate SHORT, HOLD, or LONG respectively
        - training matrix sourced from `indicators.calculate_indicators`, contains indicator values for each day of the training period
        - training target determined by if the average daily return over a lookback period exceeds a magnitude of 0.02 in either direction
    - `testPolicy`
        - queries the Random Forest using technical indicators derived from the testing period to determine trade recommendations
        - returns `trades_df` containing the number of shares (negative to indicate SHORT positions) to be traded, indexed by date

Ancillary files:
    - `BagLearner.py` : Trains and queries a Random Forest learner comprised of `RTLearner`s
    - `RTLearner.py`: Trains and queries a Random Tree Learner
    - `marketsimcode.py`: Accepts a dataframe of trades and returns resulting daily portfolio values
    - `experiment1.py`: Runs Experiment 1, an in-sample comparison of the benchmark and the Strategy and Manual learners
    - `experiment2.py`: Runs Experiment 2, an in-sample comparison of Strategy Learners at varying `impact` levels


### Usage ###
`testproject.py`:

- `main` generates all plots used in the report; pass `verbose=True` for portfolio statistics included in report tables

`ManualStrategy.py`:

    import ManualStrategy as ms
    strategy = ms.ManualStrategy()
    trades_df = strategy.testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31))

`StrategyLearner.py`:

    import StrategyLearner as sl
    learner = sl.StrategyLearner(verbose = False, impact = 0.0, commission=0.0)
    learner.add_evidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))
    trades_df = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31))
