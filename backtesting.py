import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtesting(tickers, weights, benchmark_ticker='^GSPC', global_etf_ticker='ACWI', start='2015-01-01', end='2023-01-01'):
    # Data download
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    benchmark_data = yf.download(benchmark_ticker, start=start, end=end)['Adj Close']
    global_etf_data = yf.download(global_etf_ticker, start=start, end=end)['Adj Close']

    # Daily return calc...
    daily_returns = data.pct_change().dropna()
    benchmark_returns = benchmark_data.pct_change().dropna()
    global_etf_returns = global_etf_data.pct_change().dropna()

    # Portfolio returns
    portfolio_returns = (daily_returns * weights).sum(axis=1)

    # Cumulative retruns
    cumulative_returns_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_returns_benchmark = (1 + benchmark_returns).cumprod()
    cumulative_returns_global_etf = (1 + global_etf_returns).cumprod()

    # Check if Benchmark data have correct format
    if isinstance(cumulative_returns_benchmark, pd.DataFrame):
        cumulative_returns_benchmark = cumulative_returns_benchmark.squeeze()
    if isinstance(cumulative_returns_global_etf, pd.DataFrame):
        cumulative_returns_global_etf = cumulative_returns_global_etf.squeeze()

    # Performance Metrics
    annualized_return = cumulative_returns_portfolio.iloc[-1]**(1 / (len(cumulative_returns_portfolio) / 252)) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility

    # Performance Metrics for benchmarks
    annualized_return_benchmark = cumulative_returns_benchmark.iloc[-1]**(1 / (len(cumulative_returns_benchmark) / 252)) - 1
    annualized_volatility_benchmark = benchmark_returns.std().item() * np.sqrt(252)
    sharpe_ratio_benchmark = annualized_return_benchmark / annualized_volatility_benchmark

    # Performance Metrics for benchmarks
    annualized_return_global_etf = cumulative_returns_global_etf.iloc[-1]**(1 / (len(cumulative_returns_global_etf) / 252)) - 1
    annualized_volatility_global_etf = global_etf_returns.std().item() * np.sqrt(252)  # Použijeme `.item()` pro získání float hodnoty
    sharpe_ratio_global_etf = annualized_return_global_etf / annualized_volatility_global_etf

    # Print
    print("Portfolio Performance:")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")

    print("Benchmark Performance (S&P 500):")
    print(f"Annualized Return: {annualized_return_benchmark:.2%}")
    print(f"Annualized Volatility: {annualized_volatility_benchmark:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_benchmark:.2f}\n")

    print("Global ETF Performance (ACWI):")
    print(f"Annualized Return: {annualized_return_global_etf:.2%}")
    print(f"Annualized Volatility: {annualized_volatility_global_etf:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio_global_etf:.2f}\n")

    # Graph
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns_portfolio, label='Portfolio', color='blue')
    plt.plot(cumulative_returns_benchmark, label='S&P 500', color='orange', linestyle='--')
    plt.plot(cumulative_returns_global_etf, label='Global ETF (ACWI)', color='green', linestyle=':')
    plt.title('Cumulative Returns: Portfolio vs Benchmarks')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()

    return annualized_return, annualized_volatility, sharpe_ratio
