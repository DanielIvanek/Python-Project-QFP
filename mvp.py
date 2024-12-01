import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def run_mvp(tickers, start='2015-01-01', end='2023-01-01'):
    # Stažení historických dat
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Funkce pro výpočet rizika portfolia
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Nastavení omezení a počátečního odhadu vah
    num_assets = len(tickers)
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Součet vah musí být 1
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))  # Váhy musí být mezi 0 a 1
    initial_weights = [1 / num_assets] * num_assets  # Počáteční váhy jsou rovnoměrně rozložené

    # Optimalizace portfolia
    result = minimize(
        portfolio_volatility,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    # Výpočet výsledků
    optimal_weights = result.x
    portfolio_volatility_min = portfolio_volatility(optimal_weights, cov_matrix)
    portfolio_return_min = np.sum(mean_returns * optimal_weights) * 252  # Roční výnos
    sharpe_ratio_min = portfolio_return_min / portfolio_volatility_min  # Sharpe ratio (s nulovou risk-free sazbou)

    # Zobrazení výsledků
    print("Optimal Portfolio Weights (MVP):")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
    print(f"\nExpected Annual Return: {portfolio_return_min:.2%}")
    print(f"Annual Volatility: {portfolio_volatility_min:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_min:.2f}")

    # Vykreslení grafu vah portfolia
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, optimal_weights, color='skyblue')
    plt.title('Minimum Variance Portfolio Weights')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.grid()
    plt.show()

    return optimal_weights, portfolio_return_min, portfolio_volatility_min, sharpe_ratio_min
