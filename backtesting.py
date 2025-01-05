from scipy.optimize import minimize
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def rolling_window_backtesting(tickers, window_size=3, test_period=1, start='2015-01-01', end='2023-01-01'):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    daily_returns = data.pct_change(fill_method=None).dropna()

    cumulative_returns_out_of_sample = []
    cumulative_returns_in_sample = []
    timestamps = []

    for i in range(0, len(daily_returns) - (window_size + test_period) * 252, 252):
        # Rozdělení na trénovací a testovací část
        train_data = daily_returns.iloc[i:i + window_size * 252]
        test_data = daily_returns.iloc[i + window_size * 252:i + (window_size + test_period) * 252]

        # Odhad parametrů na trénovacím období
        mean_returns = train_data.mean()
        cov_matrix = train_data.cov()

        # Optimalizace portfolia na základě trénovacích dat
        num_assets = len(tickers)
        initial_weights = [1 / num_assets] * num_assets
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

        def portfolio_volatility(weights, cov_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def objective_function(weights, mean_returns, cov_matrix):
            return -np.sum(mean_returns * weights) / portfolio_volatility(weights, cov_matrix)

        result = minimize(
            objective_function,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x

        # Výpočty na testovacích datech
        portfolio_returns_test = test_data.dot(optimal_weights)
        cumulative_return_test = (1 + portfolio_returns_test).cumprod()

        portfolio_returns_train = train_data.dot(optimal_weights)
        cumulative_return_train = (1 + portfolio_returns_train).cumprod()

        cumulative_returns_out_of_sample.append(cumulative_return_test.iloc[-1])
        cumulative_returns_in_sample.append(cumulative_return_train.iloc[-1])
        timestamps.append(test_data.index[-1])

    # Výstupní metriky
    avg_out_of_sample_return = np.mean(cumulative_returns_out_of_sample)
    avg_in_sample_return = np.mean(cumulative_returns_in_sample)

    print("Rolling Window Backtesting Results:")
    print(f"Average Out-of-Sample Return: {avg_out_of_sample_return:.2%}")
    print(f"Average In-Sample Return: {avg_in_sample_return:.2%}")

    # Grafické zobrazení kumulativních výnosů
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cumulative_returns_out_of_sample, label='Out-of-Sample Returns', marker='o')
    plt.plot(timestamps, cumulative_returns_in_sample, label='In-Sample Returns', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.title('Out-of-Sample vs In-Sample Performance (Rolling Window)')
    plt.grid()
    plt.show()
