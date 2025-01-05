import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt

# Parametry
tickers = ['CVS', 'BABA', 'BTC-USD', 'ETH-USD']
end_date = datetime.today()
start_date = end_date - timedelta(days=5 * 365)  # 10 let historických dat
training_years = 3  # Délka trénovacího okna (roky)
test_years = 1  # Délka testovacího okna (roky)

# Stáhnutí dat
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_df[ticker] = data['Adj Close']

# Logaritmické výnosy
log_returns = np.log(adj_close_df / adj_close_df.shift(1))
log_returns = log_returns.dropna()

# Funkce pro výpočet metrik
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Rolling window backtesting
rolling_results = []
risk_free_rate = 0.02  # Konstantní sazba, můžete nahradit dynamickou hodnotou z FRED API

for start_idx in range(0, len(log_returns) - (training_years + test_years) * 252, 252):
    # Trénovací a testovací data
    train_data = log_returns.iloc[start_idx:start_idx + training_years * 252]
    test_data = log_returns.iloc[start_idx + training_years * 252:start_idx + (training_years + test_years) * 252]

    # Odhad parametrů na trénovacích datech
    mean_returns = train_data.mean()
    cov_matrix = train_data.cov()

    # Tisk kovarianční matice
    print(f"\nCovariance Matrix for Training Period {train_data.index[0].date()} to {train_data.index[-1].date()}:\n")
    print(cov_matrix)

    # Heatmapa kovarianční matice
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"Covariance Matrix: {train_data.index[0].date()} to {train_data.index[-1].date()}")
    plt.xlabel("Ticker")
    plt.ylabel("Ticker")
    plt.show()

    # Optimalizace portfolia
    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    bounds = [(0, 0.4) for _ in range(len(tickers))]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    result = minimize(
        neg_sharpe_ratio,
        initial_weights,
        args=(train_data, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    # Výpočet výkonu na testovacích datech
    portfolio_returns_test = test_data.dot(optimal_weights)
    cumulative_return_test = (1 + portfolio_returns_test).cumprod()

    # Výpočet metrik
    annualized_return = cumulative_return_test.iloc[-1]**(1 / test_years) - 1
    annualized_volatility = portfolio_returns_test.std() * np.sqrt(252)
    sharpe = sharpe_ratio(optimal_weights, test_data, cov_matrix, risk_free_rate)

    rolling_results.append({
        'Start Date': test_data.index[0],
        'End Date': test_data.index[-1],
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe
    })

# Výsledky
results_df = pd.DataFrame(rolling_results)
print(results_df)

# Grafické zobrazení
plt.figure(figsize=(10, 6))
plt.plot(results_df['Start Date'], results_df['Annualized Return'], label='Annualized Return', marker='o')
plt.plot(results_df['Start Date'], results_df['Annualized Volatility'], label='Annualized Volatility', marker='x')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Test Period Start Date')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Rolling Window Backtesting Results')
plt.grid()
plt.show()
