import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo(data, weights, num_simulations=1000, num_days=252, initial_portfolio_value=100000):
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    simulated_portfolio_values = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        simulated_daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        simulated_portfolio_returns = np.dot(simulated_daily_returns, weights)
        simulated_portfolio_values[:, i] = initial_portfolio_value * (1 + simulated_portfolio_returns).cumprod()

    plt.figure(figsize=(12, 6))
    for i in range(num_simulations):
        plt.plot(simulated_portfolio_values[:, i], color='blue', alpha=0.05)
    plt.title('Monte Carlo Simulation of Portfolio Returns')
    plt.xlabel('Days')
    plt.ylabel('Portfolio Value')
    plt.axhline(y=initial_portfolio_value, color='red', linestyle='--', label='Initial Value')
    plt.legend()
    plt.grid()
    plt.show()

    ending_values = simulated_portfolio_values[-1, :]
    mean_ending_value = np.mean(ending_values)
    median_ending_value = np.median(ending_values)
    percentile_5 = np.percentile(ending_values, 5)
    percentile_95 = np.percentile(ending_values, 95)

    return mean_ending_value, median_ending_value, percentile_5, percentile_95
