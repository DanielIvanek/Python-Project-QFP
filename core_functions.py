# core_functions.py
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt


class PortfolioOptimizer:
    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)
        self.risk_free_rate = 0.02  # Fallback value

    def download_data(self, tickers, start_date, end_date):
        """Stáhne data pro multi-asset portfolio s kontrolou kvality dat"""
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        data = data.dropna(axis=1, how='all')  # Odstraní sloupce bez dat
        return data

    def calculate_returns(self, price_data, freq='D'):
        """Vypočítá logaritmické výnosy s možností různých frekvencí"""
        if freq != 'D':
            price_data = price_data.resample(freq).last()
        return np.log(price_data / price_data.shift(1)).dropna()

    def get_risk_free_rate(self, target_date):
        """Získá dynamickou bezrizikovou sazbu s historickou perspektivou"""
        try:
            series = self.fred.get_series('GS10', target_date.strftime('%Y-%m-%d'))
            if series.empty:
                series = self.fred.get_series('GS10')
                series = series[series.index <= target_date]
            self.risk_free_rate = series.iloc[-1] / 100 if not series.empty else 0.02
        except Exception as e:
            print(f"Chyba FRED API: {str(e)}, použita fallback hodnota 2%")
        return self.risk_free_rate

    def optimize_portfolio(self, returns, risk_free_rate=0.02,
                           asset_constraints=None, lambda_reg=0.1):
        """
        Pokročilá optimalizace portfolia s:
        - Sektorovými omezeními
        - Kombinovanou L1/L2 regularizací
        - Dynamickými hranicemi pro třídy aktiv
        """
        tickers = returns.columns.tolist()
        cov_matrix = returns.cov() * 252

        # Defaultní omezení
        if not asset_constraints:
            asset_constraints = {
                'tech': {'assets': ['XLK', 'QQQ'], 'max': 0.3},
                'bonds': {'assets': ['BND'], 'min': 0.1}
            }

        # Příprava omezení
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # Přidání sektorových omezení
        for sector, params in asset_constraints.items():
            indices = [tickers.index(a) for a in params['assets'] if a in tickers]
            if indices:
                if 'max' in params:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, i=indices, m=params['max']: m - np.sum(w[i])
                    })
                if 'min' in params:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, i=indices, m=params['min']: np.sum(w[i]) - m
                    })

        # Dynamické hranice pro různé třídy aktiv
        bounds = []
        for t in tickers:
            if t in ['BND']:  # Dluhopisy
                bounds.append((0.05, 0.3))
            elif t in ['GLD']:  # Komodity
                bounds.append((0.05, 0.2))
            else:  # Akcie
                bounds.append((0.05, 0.25))

        def objective(w):
            port_return = np.sum(returns.mean() * w) * 252  # Roční výnos
            port_vol = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(252)  # Annualizovaná volatilita
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else 0
            return -sharpe  # Negativní Sharpe ratio pro minimalizaci

        def objective(w):
            port_return = np.sum(returns.mean() * w) * 252
            port_vol = np.sqrt(w.T @ cov_matrix @ w)
            sharpe = (port_return - risk_free_rate) / port_vol if port_vol != 0 else 0

            # Kombinovaná regularizace
            l1_penalty = lambda_reg * np.sum(np.abs(w - 0.1))
            l2_penalty = lambda_reg * np.sum(w ** 2)

            return -sharpe + l1_penalty + l2_penalty

        # Optimalizace
        result = minimize(
            objective,
            x0=np.array([1 / len(tickers)] * len(tickers)),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        return result.x

    def analyze_portfolio(self, weights, returns, risk_free_rate):
        """Komplexní analýza portfolia s rizikovými metrikami"""
        metrics = {}
        metrics['Return'] = np.sum(returns.mean() * weights) * 252
        metrics['Volatility'] = np.sqrt(weights.T @ returns.cov() * 252 @ weights)
        metrics['Sharpe'] = (metrics['Return'] - risk_free_rate) / metrics['Volatility']

        # Výpočet Value at Risk (95% úroveň důvěryhodnosti)
        portfolio_returns = returns @ weights
        metrics['VaR_95'] = np.percentile(portfolio_returns, 5)

        return metrics

    def plot_correlation_matrix(self, returns):
        """Vykreslí heatmapu korelací"""
        plt.figure(figsize=(12, 8))
        sns.heatmap(returns.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Korelační matice aktiv')
        plt.show()


if __name__ == "__main__":
    # Příklad použití
    optimizer = PortfolioOptimizer(fred_api_key='73bf9caaa3bc8dcde51bd1d08eaf83d9')
    #tickers = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD', 'XLK']
    tickers = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD', 'XLK', 'BTC-USD']
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    # Stažení a příprava dat
    price_data = optimizer.download_data(tickers, start_date, end_date)
    returns = optimizer.calculate_returns(price_data)

    # Tréninkové/testovací rozdělení
    split_date = end_date - timedelta(days=2 * 365)
    train_returns = returns[returns.index <= split_date]
    test_returns = returns[returns.index > split_date]

    # Optimalizace
    risk_free_rate = optimizer.get_risk_free_rate(train_returns.index[-1])
    weights = optimizer.optimize_portfolio(train_returns, risk_free_rate)

    # Analýza
    train_metrics = optimizer.analyze_portfolio(weights, train_returns, risk_free_rate)
    test_metrics = optimizer.analyze_portfolio(weights, test_returns, risk_free_rate)

    # Výsledky
    print("Optimální alokace:")
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker}: {weight:.2%}")

    print("\nTréninkové období:")
    print(
        f"Výnos: {train_metrics['Return']:.2%} | Volatilita: {train_metrics['Volatility']:.2%} | Sharpe: {train_metrics['Sharpe']:.2f}")

    print("\nTestovací období:")
    print(
        f"Výnos: {test_metrics['Return']:.2%} | Volatilita: {test_metrics['Volatility']:.2%} | Sharpe: {test_metrics['Sharpe']:.2f}")

    # Vizualizace
    optimizer.plot_correlation_matrix(returns)

