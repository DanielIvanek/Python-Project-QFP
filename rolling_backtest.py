# rolling_backtest.py

import pandas as pd
from datetime import datetime, timedelta
from core_functions import PortfolioOptimizer
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt


class Backtester:

    def __init__(self, fred_api_key):
        self.optimizer = PortfolioOptimizer(fred_api_key)

    def run_backtest(self, tickers, start_date, end_date,
                     train_years=3, test_years=1,
                     asset_constraints=None, lambda_reg=0.1):

        price_data = self.optimizer.download_data(tickers, start_date, end_date)
        returns = self.optimizer.calculate_returns(price_data)

        results = []
        current_date = start_date + timedelta(days=train_years * 365)

        while current_date + timedelta(days=test_years * 365) <= end_date:
            train_mask = returns.index <= current_date
            test_mask = (returns.index > current_date) & \
                        (returns.index <= current_date + timedelta(days=test_years * 365))

            train_returns = returns[train_mask]
            test_returns = returns[test_mask]

            if len(train_returns) < 100 or len(test_returns) < 20:
                current_date += timedelta(days=test_years * 365)
                continue

            try:
                risk_free_rate = self.optimizer.get_risk_free_rate(train_returns.index[-1])
                weights = self.optimizer.optimize_portfolio(
                    train_returns,
                    risk_free_rate,
                    asset_constraints=asset_constraints,
                    lambda_reg=lambda_reg
                )

                train_metrics = self.optimizer.analyze_portfolio(weights, train_returns, risk_free_rate)
                test_metrics = self.optimizer.analyze_portfolio(weights, test_returns, risk_free_rate)

                result = {
                    'Start Date': current_date.strftime('%Y-%m-%d'),
                    'End Date': (current_date + timedelta(days=test_years * 365)).strftime('%Y-%m-%d'),
                    **{ticker: weight for ticker, weight in zip(tickers, weights)},
                    'Train Sharpe': train_metrics['Sharpe'],
                    'Test Sharpe': test_metrics['Sharpe'],
                    'Test Return': test_metrics['Return'],
                    'Test Volatility': test_metrics['Volatility'],
                    'Test VaR 95%': test_metrics['VaR_95']
                }
                results.append(result)

            except Exception as e:
                print(f"Chyba v okně {current_date}: {str(e)}")

            current_date += timedelta(days=test_years * 365)

        return pd.DataFrame(results)

    def plot_cumulative_returns(self, results_df, initial_investment=100000):

        # Výpočet kumulativních výnosů
        returns = results_df['Test Return'] / 252  # Denní výnosy
        cumulative_returns = (1 + returns).cumprod()
        portfolio_value = initial_investment * cumulative_returns

        # Benchmark (S&P 500)
        spy = self.optimizer.download_data(['SPY'], results_df['Start Date'].min(), results_df['End Date'].max())
        spy_returns = self.optimizer.calculate_returns(spy).mean(axis=1)
        spy_cumulative = (1 + spy_returns).cumprod()
        spy_value = initial_investment * spy_cumulative

        # Vykreslení
        plt.figure(figsize=(12, 6))
        portfolio_value.plot(label='Optimalizované portfolio')
        spy_value.plot(label='Benchmark (SPY)', linestyle='--')
        plt.title(f'Vývoj portfolia od počáteční investice ${initial_investment:,.0f}')
        plt.ylabel('Hodnota portfolia ($)')
        plt.xlabel('Datum')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_weights_evolution(self, results_df):
        weight_cols = [col for col in results_df.columns if col not in [
            'Start Date', 'End Date', 'Train Sharpe', 'Test Sharpe',
            'Test Return', 'Test Volatility', 'Test VaR 95%'
        ]]

        # Připravíme data pro stacked area chart
        df = results_df.set_index('End Date')[weight_cols]
        df.index = pd.to_datetime(df.index)

        plt.figure(figsize=(12, 6))
        plt.stackplot(df.index, df.values.T, labels=weight_cols)
        plt.title('Vývoj alokací v čase')
        plt.ylabel('Alokace (%)')
        plt.xlabel('Datum')
        plt.legend(loc='upper left')
        plt.ylim(0, 1)
        plt.show()

    def plot_risk_return_tradeoff(self, results_df):

        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Test Volatility'],
                    results_df['Test Return'],
                    c=results_df['Test Sharpe'],
                    cmap='viridis',
                    alpha=0.7)
        plt.colorbar(label='Sharpeho poměr')
        plt.title('Risk-Return Tradeoff')
        plt.xlabel('Roční volatilita')
        plt.ylabel('Roční výnos')
        plt.grid(True)

        max_sharpe_idx = results_df['Test Sharpe'].idxmax()
        plt.annotate(f"Nejlepší Sharpe: {results_df.loc[max_sharpe_idx, 'Test Sharpe']:.2f}",
                     xy=(results_df.loc[max_sharpe_idx, 'Test Volatility'],
                         results_df.loc[max_sharpe_idx, 'Test Return']),
                     xytext=(0.3, 0.7), textcoords='axes fraction',
                     arrowprops=dict(facecolor='red', shrink=0.05))
        plt.show()

    def plot_drawdown(self, results_df):

        weight_cols = [col for col in results_df.columns if col not in [
            'Start Date', 'End Date', 'Train Sharpe', 'Test Sharpe',
            'Test Return', 'Test Volatility', 'Test VaR 95%'
        ]]

        drawdowns = []
        for _, row in results_df.iterrows():
            returns = self.optimizer.calculate_returns(
                self.optimizer.download_data(
                    weight_cols,
                    row['Start Date'],
                    row['End Date']
                )
            )
            valid_tickers = [ticker for ticker in weight_cols if ticker in returns.columns]
            returns = returns[valid_tickers]
            weights = row[valid_tickers]
            portfolio_returns = returns @ weights

            cumulative = (1 + portfolio_returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative / peak - 1).min()
            drawdowns.append(drawdown)

        plt.figure(figsize=(12, 6))
        plt.bar(results_df['End Date'], drawdowns, color='red')
        plt.title('Maximální drawdown v každém období')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Datum')
        plt.ylim(-0.5, 0)
        plt.show()

    def generate_report(self, results_df):

        if results_df.empty:
            print("Žádné výsledky k reportování")
            return

        weight_cols = [col for col in results_df.columns if col not in [
            'Start Date', 'End Date', 'Train Sharpe', 'Test Sharpe',
            'Test Return', 'Test Volatility', 'Test VaR 95%'
        ]]

        display_df = results_df.copy()
        for col in weight_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

        print("\n" + "=" * 50)
        print("Výsledky rolling window backtestu:")
        print("=" * 50)
        print(tabulate(display_df, headers='keys', tablefmt='pretty', floatfmt=".4f"))

        print("\n" + "=" * 50)
        print("Statistické shrnutí:")
        print("=" * 50)
        stats = results_df.agg({
            'Train Sharpe': ['mean', 'std', 'min', 'max'],
            'Test Sharpe': ['mean', 'std', 'min', 'max'],
            'Test Return': ['mean', 'std'],
            'Test Volatility': ['mean', 'std'],
            'Test VaR 95%': ['mean', 'min']
        })
        print(tabulate(stats, headers='keys', tablefmt='pretty', floatfmt=".4f"))

        # Analýza stability alokací
        print("\n" + "=" * 50)
        print("Stabilita alokací (nejčastější váhy):")
        print("=" * 50)
        for col in weight_cols:
            counts = results_df[col].value_counts(normalize=True).head(3)
            print(f"\n{col}:")
            for value, freq in counts.items():
                print(f"{value:.2%} se vyskytlo v {freq:.2%} případů")

        # Heatmap correlation
        plt.figure(figsize=(12, 8))
        sns.heatmap(results_df[weight_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Korelace mezi alokacemi')
        plt.show()

        # Nové vizualizace
        self.plot_cumulative_returns(results_df)
        self.plot_weights_evolution(results_df)
        self.plot_risk_return_tradeoff(results_df)
        self.plot_drawdown(results_df)

    def save_results(self, results_df, filename):

        results_df.to_csv(filename, index=False)
        print(f"\nVýsledky uloženy do {filename}")


if __name__ == "__main__":
    # Konfigurace
    FRED_API_KEY = '73bf9caaa3bc8dcde51bd1d08eaf83d9'
    TICKERS = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD', 'XLK', 'BTC-USD']
    ASSET_CONSTRAINTS = {
        'tech': {'assets': ['XLK', 'QQQ'], 'max': 0.4},
        'bonds': {'assets': ['BND'], 'min': 0.1, 'max': 0.3}
    }

    backtester = Backtester(FRED_API_KEY)

    results = backtester.run_backtest(
        tickers=TICKERS,
        start_date=datetime(2013, 1, 1),
        end_date=datetime(2024, 1, 1),
        train_years=2,
        test_years=1,
        asset_constraints=ASSET_CONSTRAINTS,
        lambda_reg=0.15
    )

    backtester.generate_report(results)
    backtester.save_results(results, "backtest_results.csv")
