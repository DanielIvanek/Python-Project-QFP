import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Definujte akcie, váhy a benchmarky
tickers = ['CVS', 'BABA', 'EVO.ST', 'TUI1.DE']  # Akcie v portfoliu
weights = [0.25, 0.25, 0.25, 0.25]  # Váhy akcií (musí dávat dohromady 1)
benchmark_ticker = '^GSPC'  # Benchmark: S&P 500
global_etf_ticker = 'ACWI'  # Globální ETF

# 2. Stáhněte historická data
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Adj Close']
benchmark_data = yf.download(benchmark_ticker, start='2015-01-01', end='2023-01-01')['Adj Close']
global_etf_data = yf.download(global_etf_ticker, start='2015-01-01', end='2023-01-01')['Adj Close']

# 3. Vypočítejte denní výnosy
daily_returns = data.pct_change(fill_method=None).dropna()
benchmark_returns = benchmark_data.pct_change(fill_method=None).dropna()
global_etf_returns = global_etf_data.pct_change(fill_method=None).dropna()

# 4. Výpočet portfoliového výnosu
portfolio_returns = (daily_returns * weights).sum(axis=1)

# 5. Kumulativní návratnost portfolia a benchmarků
cumulative_returns_portfolio = (1 + portfolio_returns).cumprod()
cumulative_returns_benchmark = (1 + benchmark_returns).cumprod()
cumulative_returns_global_etf = (1 + global_etf_returns).cumprod()

# Ujistěte se, že benchmark data mají správný formát
if isinstance(cumulative_returns_benchmark, pd.DataFrame):
    cumulative_returns_benchmark = cumulative_returns_benchmark.squeeze()
if isinstance(cumulative_returns_global_etf, pd.DataFrame):
    cumulative_returns_global_etf = cumulative_returns_global_etf.squeeze()

# 6. Výkonnostní metriky pro portfolio
annualized_return_portfolio = cumulative_returns_portfolio.iloc[-1]**(1 / (len(cumulative_returns_portfolio) / 252)) - 1
annualized_volatility_portfolio = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio_portfolio = annualized_return_portfolio / annualized_volatility_portfolio

## 7. Výkonnostní metriky pro benchmark (S&P 500)
if isinstance(benchmark_returns, pd.DataFrame):
    benchmark_returns = benchmark_returns.squeeze()  # Převést na Series, pokud má jen jeden sloupec
if isinstance(global_etf_returns, pd.DataFrame):
    global_etf_returns = global_etf_returns.squeeze()  # Převést na Series, pokud má jen jeden sloupec

# Výpočty metrik pro benchmark (S&P 500)
annualized_return_benchmark = cumulative_returns_benchmark.iloc[-1]**(1 / (len(cumulative_returns_benchmark) / 252)) - 1
annualized_volatility_benchmark = benchmark_returns.std(ddof=1) * np.sqrt(252)  # ddof=1 pro standardní odchylku vzorku
sharpe_ratio_benchmark = annualized_return_benchmark / annualized_volatility_benchmark

# Výpočty metrik pro globální ETF
annualized_return_global_etf = cumulative_returns_global_etf.iloc[-1]**(1 / (len(cumulative_returns_global_etf) / 252)) - 1
annualized_volatility_global_etf = global_etf_returns.std(ddof=1) * np.sqrt(252)
sharpe_ratio_global_etf = annualized_return_global_etf / annualized_volatility_global_etf

# 9. Zobrazení výsledků
print("Benchmark Performance (S&P 500):")
print(f"Annualized Return: {annualized_return_benchmark:.2%}")
print(f"Annualized Volatility: {annualized_volatility_benchmark:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_benchmark:.2f}\n")

print("Global ETF Performance (ACWI):")
print(f"Annualized Return: {annualized_return_global_etf:.2%}")
print(f"Annualized Volatility: {annualized_volatility_global_etf:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio_global_etf:.2f}")


# 10. Graf kumulativní návratnosti
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