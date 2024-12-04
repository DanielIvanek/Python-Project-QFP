from monte_carlo_simulation import run_monte_carlo
from backtesting import run_backtesting
import yfinance as yf
from markowitz_optimization import run_markowitz_optimization
from mvp import run_mvp


tickers = ['CVS', 'BABA', 'EVO.ST', 'TUI1.DE']
weights = [0.25 , 0.25, 0.25, 0.25]

# Montecarlo data download
data = yf.download(tickers, start='2015-01-01', end='2023-01-01')['Adj Close']

# Monte Carlo simulation
print("Monte Carlo Simulation:")
mean_ending_value, median_ending_value, fifth_percentile, ninety_fifth_percentile = run_monte_carlo(
    data, weights, num_simulations=1000, num_days=252, initial_portfolio_value=100000
)
print(f"Mean Ending Value: {mean_ending_value:.2f}")
print(f"Median Ending Value: {median_ending_value:.2f}")
print(f"5th Percentile: {fifth_percentile:.2f}")
print(f"95th Percentile: {ninety_fifth_percentile:.2f}")

# Backtesting
print("\nBacktesting:")
annualized_return, annualized_volatility, sharpe_ratio = run_backtesting(
    tickers, weights, benchmark_ticker='^GSPC', global_etf_ticker='ACWI'
)

# Markowitz Optimization
print("\nMarkowitz Portfolio Optimization:")
run_markowitz_optimization(tickers)

# Minimum Variance Portfolio (MVP)
print("\nMinimum Variance Portfolio (MVP):")
run_mvp(tickers)
