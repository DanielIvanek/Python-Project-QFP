import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def run_markowitz_optimization(tickers, start='2015-01-01', end='2023-01-01', risk_free_rate=0.01):
    # Download data
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Volatility
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Optimalizace Sharpe Ratio s penalizací koncentrace

    # Optimalization Sharpe ration and penalize concentration
    def objective_function(weights, mean_returns, cov_matrix, risk_free_rate):
        sharpe_ratio = -(np.sum(mean_returns * weights) * 252 - risk_free_rate) / portfolio_volatility(weights, cov_matrix)
        concentration_penalty = np.sum(np.square(weights))
        return sharpe_ratio + 0.1 * concentration_penalty

    # Constraints and weights
    num_assets = len(tickers)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = tuple((0.05, 0.5) for _ in range(num_assets))
    initial_weights = [1 / num_assets] * num_assets

    # Optimization
    result = minimize(
        objective_function,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    portfolio_return = np.sum(mean_returns * optimal_weights) * 252
    portfolio_volatility = portfolio_volatility(optimal_weights, cov_matrix)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Results
    print("Optimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"\nExpected Annual Return: {portfolio_return:.2%}")
    print(f"Annual Volatility: {portfolio_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Graph
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights, color='skyblue')
    plt.title('Optimal Portfolio Weights (With Diversification Constraints)')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.grid()
    plt.show()

    return portfolio_return, portfolio_volatility, sharpe_ratio
