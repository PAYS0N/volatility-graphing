"""
Historical Stock Option Implied Volatility Analyzer
Uses DoltHub's free options dataset (post-no-preference/options)
The dataset already includes calculated IV - we retrieve and visualize it
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class OptionIVAnalyzer:
    def __init__(self):
        """Initialize DoltHub API connection"""
        self.base_url = "https://www.dolthub.com/api/v1alpha1"
        self.owner = "post-no-preference"
        self.repo = "options"
        self.branch = "master"
        
    def query_dolthub(self, sql_query):
        """
        Execute SQL query against DoltHub repository
        """
        url = f"{self.base_url}/{self.owner}/{self.repo}/{self.branch}"
        
        try:
            response = requests.get(url, params={"q": sql_query}, headers={ "authorization": "token dhat.v1.gpikpgdn8q7ob73ke8rh2j7j8sq8noiiun69e2cpod4vgnqft1h0" },)
            response.raise_for_status()
            data = response.json()
            
            if data.get('query_execution_status') == 'Success' and 'rows' in data:
                df = pd.DataFrame(data['rows'])
                # Convert date columns
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                if 'expiration' in df.columns:
                    df['expiration'] = pd.to_datetime(df['expiration'])
                # Convert numeric columns
                numeric_cols = ['strike', 'bid', 'ask', 'vol', 'delta', 'gamma', 'theta', 'vega', 'rho']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
            else:
                print(f"Query failed: {data.get('query_execution_message', 'Unknown error')}")
                return pd.DataFrame()
                
        except requests.exceptions.RequestException as e:
            print(f"Error querying DoltHub: {e}")
            return pd.DataFrame()
    
    def find_nearest_strike(self, ticker, expiration_date, target_strike):
        """
        Find the nearest available strike price for given expiration
        """
        query = f"""
        SELECT DISTINCT strike 
        FROM option_chain 
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{expiration_date}'
        ORDER BY strike;
        """
        
        print("Querying dolthub to find closest strike")
        strikes_df = self.query_dolthub(query)
        
        if strikes_df.empty:
            return None
        
        strikes = strikes_df['strike'].values
        nearest_strike = min(strikes, key=lambda x: abs(x - target_strike))
        
        return nearest_strike
    
    def get_option_iv_history(self, ticker, expiration_date, strike, option_type, days_back=90):
        """
        Fetch historical implied volatility data from DoltHub
        
        Parameters:
        -----------
        ticker: str - Stock ticker symbol (e.g., 'AAPL')
        expiration_date: str - Expiration date in 'YYYY-MM-DD' format
        strike: float - Strike price (will find nearest available)
        option_type: str - 'call' or 'put'
        days_back: int - Number of days before expiration to retrieve
        
        Returns:
        --------
        DataFrame with IV history
        """
        # Convert option type to match database format
        call_put = 'Call' if option_type.lower() == 'call' else 'Put'
        
        # Calculate date range
        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        start_date = (exp_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # First, find the nearest available strike
        print(f"Finding nearest strike to ${strike}...")
        actual_strike = self.find_nearest_strike(ticker, expiration_date, strike)
        
        if actual_strike is None:
            print(f"No options found for {ticker} expiring on {expiration_date}")
            return None, None
        
        print(f"Using strike: ${actual_strike:.2f}")
        
        # Query for option data
        query = f"""
        SELECT date, act_symbol, expiration, strike, call_put, bid, ask, vol, 
               delta, gamma, theta, vega, rho
        FROM option_chain
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{expiration_date}'
        AND strike = {actual_strike}
        AND call_put = '{call_put}'
        AND date >= '{start_date}'
        AND date <= '{expiration_date}'
        ORDER BY date;
        """
        
        print(f"Querying DoltHub for {ticker} {call_put} data...")
        df = self.query_dolthub(query)
        
        if df.empty:
            print("No data found")
            return None, actual_strike
        
        # Calculate mid price
        df['option_price'] = (df['bid'] + df['ask']) / 2
        
        # Convert IV from decimal to percentage
        df['iv_percent'] = df['vol'] * 100
        
        # Calculate days to expiration
        df['days_to_expiration'] = (df['expiration'] - df['date']).dt.days
        
        return df, actual_strike
    
    def get_stock_price_history(self, ticker, start_date, end_date):
        """
        Fetch historical stock prices using yfinance
        """
        try:
            print(f"Fetching stock price history for {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                return pd.DataFrame()
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            hist = hist.rename(columns={'Date': 'date', 'Close': 'stock_price'})
            hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)
            
            return hist[['date', 'stock_price']]
        except Exception as e:
            print(f"Error fetching stock prices: {e}")
            return pd.DataFrame()
    
    def analyze_option(self, ticker, expiration_date, strike, option_type, days_back=90):
        """
        Main function to analyze option IV over time
        
        Parameters:
        -----------
        ticker: str - e.g., 'AAPL'
        expiration_date: str - 'YYYY-MM-DD' format
        strike: float - Target strike price
        option_type: str - 'call' or 'put'
        days_back: int - Days before expiration to analyze (default 90)
        """
        print(f"\n{'='*70}")
        print(f"Analyzing {ticker} {option_type.upper()} Option")
        print(f"Target Strike: ${strike}, Expiration: {expiration_date}")
        print(f"{'='*70}\n")
        
        # Get option IV data
        option_data, actual_strike = self.get_option_iv_history(
            ticker, expiration_date, strike, option_type, days_back
        )
        
        if option_data is None or option_data.empty:
            print("No option data available")
            return None
        
        print(f"\nRetrieved {len(option_data)} days of data")
        print(f"Date range: {option_data['date'].min().date()} to {option_data['date'].max().date()}")
        print(f"IV range: {option_data['iv_percent'].min():.2f}% to {option_data['iv_percent'].max():.2f}%")
        print(f"Average IV: {option_data['iv_percent'].mean():.2f}%")
        
        # Get stock price history
        start_date = option_data['date'].min()
        end_date = option_data['date'].max()
        stock_data = self.get_stock_price_history(ticker, start_date, end_date)
        
        # Merge option and stock data
        if not stock_data.empty:
            merged_data = pd.merge(option_data, stock_data, on='date', how='left')
        else:
            merged_data = option_data.copy()
            merged_data['stock_price'] = np.nan
            print("\nWarning: Could not retrieve stock price data")
        
        # Add strike info to return
        merged_data['actual_strike'] = actual_strike
        
        return merged_data
    
    def plot_iv_chart(self, data, ticker, target_strike, expiration_date, option_type):
        """
        Create comprehensive visualization of IV and related metrics
        """
        if data is None or data.empty:
            print("No data to plot")
            return None
        
        actual_strike = data['actual_strike'].iloc[0]
        has_stock_price = 'stock_price' in data.columns and not data['stock_price'].isna().all()
        
        # Create figure with subplots
        if has_stock_price:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0:2, :])  # IV plot (top, full width)
            ax2 = fig.add_subplot(gs[2, :])     # Stock price (middle, full width)
            ax3 = fig.add_subplot(gs[3, 0])     # Greeks 1
            ax4 = fig.add_subplot(gs[3, 1])     # Greeks 2
        else:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0:2, :])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
        
        # Plot 1: Implied Volatility
        ax1.plot(data['date'], data['iv_percent'], 
                linewidth=2.5, color='#2E86AB', marker='o', markersize=5,
                label='Implied Volatility')
        ax1.axhline(y=data['iv_percent'].mean(), color='#A23B72', 
                   linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Average: {data["iv_percent"].mean():.2f}%')
        ax1.fill_between(data['date'], data['iv_percent'], alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')
        ax1.set_title(
            f'{ticker} {option_type.upper()} Option - Implied Volatility Analysis\n'
            f'Strike: ${actual_strike:.2f} (target: ${target_strike:.2f}), '
            f'Expiration: {expiration_date}',
            fontsize=15, fontweight='bold', pad=20
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=11)
        ax1.tick_params(axis='both', labelsize=10)
        
        # Plot 2: Stock Price (if available)
        if has_stock_price:
            ax2.plot(data['date'], data['stock_price'], 
                    linewidth=2.5, color='#F18F01', marker='o', markersize=5,
                    label='Stock Price')
            ax2.axhline(y=actual_strike, color='#C73E1D', 
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Strike: ${actual_strike:.2f}')
            ax2.fill_between(data['date'], data['stock_price'], alpha=0.2, color='#F18F01')
            ax2.set_ylabel('Stock Price ($)', fontsize=13, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=11)
            ax2.tick_params(axis='both', labelsize=10)
            
            # Calculate moneyness
            data['moneyness'] = (data['stock_price'] / actual_strike - 1) * 100
        
        # Plot 3: Delta and Gamma
        ax3.plot(data['date'], data['delta'], 
                linewidth=2, color='#06A77D', marker='s', markersize=4,
                label='Delta')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(data['date'], data['gamma'], 
                     linewidth=2, color='#D62246', marker='^', markersize=4,
                     label='Gamma')
        ax3.set_ylabel('Delta', fontsize=11, fontweight='bold', color='#06A77D')
        ax3_twin.set_ylabel('Gamma', fontsize=11, fontweight='bold', color='#D62246')
        ax3.set_xlabel('Date', fontsize=10)
        ax3.tick_params(axis='y', labelcolor='#06A77D')
        ax3_twin.tick_params(axis='y', labelcolor='#D62246')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Delta & Gamma', fontsize=12, fontweight='bold')
        
        # Plot 4: Theta and Vega
        ax4.plot(data['date'], data['theta'], 
                linewidth=2, color='#9C528B', marker='o', markersize=4,
                label='Theta')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(data['date'], data['vega'], 
                     linewidth=2, color='#F4A261', marker='d', markersize=4,
                     label='Vega')
        ax4.set_ylabel('Theta', fontsize=11, fontweight='bold', color='#9C528B')
        ax4_twin.set_ylabel('Vega', fontsize=11, fontweight='bold', color='#F4A261')
        ax4.set_xlabel('Date', fontsize=10)
        ax4.tick_params(axis='y', labelcolor='#9C528B')
        ax4_twin.tick_params(axis='y', labelcolor='#F4A261')
        ax4.grid(True, alpha=0.3)
        ax4.set_title('Theta & Vega', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels
        for ax in [ax1, ax3, ax4]:
            ax.tick_params(axis='x', rotation=45)
        if has_stock_price:
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def batch_analyze(self, tickers, expiration_date, strike, option_type, days_back=90):
        """
        Analyze multiple tickers and create comparison plots
        """
        all_data = {}
        
        for ticker in tickers:
            print(f"\nProcessing {ticker}...")
            data = self.analyze_option(ticker, expiration_date, strike, option_type, days_back)
            if data is not None:
                all_data[ticker] = data
        
        if not all_data:
            print("No data collected for any ticker")
            return None
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        colors = ['#2E86AB', '#F18F01', '#06A77D', '#D62246', '#9C528B', '#F4A261', '#264653']
        
        for idx, (ticker, data) in enumerate(all_data.items()):
            color = colors[idx % len(colors)]
            ax1.plot(data['date'], data['iv_percent'], 
                    linewidth=2, marker='o', markersize=4,
                    label=ticker, color=color, alpha=0.8)
            
            if 'stock_price' in data.columns and not data['stock_price'].isna().all():
                # Normalize stock prices for comparison (% change from start)
                normalized = (data['stock_price'] / data['stock_price'].iloc[0] - 1) * 100
                ax2.plot(data['date'], normalized,
                        linewidth=2, marker='o', markersize=4,
                        label=ticker, color=color, alpha=0.8)
        
        ax1.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')
        ax1.set_title(
            f'Multi-Stock IV Comparison - {option_type.upper()} Options\n'
            f'Expiration: {expiration_date}',
            fontsize=15, fontweight='bold'
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10, ncol=2)
        
        ax2.set_ylabel('Stock Price % Change', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10, ncol=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, all_data


def main():
    """
    Main execution function with examples
    """
    
    # CONFIGURATION
    ticker = "AAPL"  # Or try: ["AAPL", "GOOGL", "AMZN", "META", "MSFT", "NVDA", "TSLA"]
    strike_price = 150.0
    expiration_date = "2019-12-20"  # Must be a date in the past (database has data from ~2018-2020)
    option_type = "call"  # 'call' or 'put'
    days_back = 90
    
    # Initialize analyzer
    analyzer = OptionIVAnalyzer()
    
    # Single ticker analysis
    print("\n" + "="*70)
    print("SINGLE TICKER ANALYSIS")
    print("="*70)
    
    data = analyzer.analyze_option(
        ticker=ticker,
        expiration_date=expiration_date,
        strike=strike_price,
        option_type=option_type,
        days_back=days_back
    )
    
    if data is not None:
        # Save data
        filename = f"{ticker}_{option_type}_{strike_price}_{expiration_date}_IV_data.csv"
        data.to_csv(filename, index=False)
        print(f"\nData saved to: {filename}")
        
        # Create plot
        fig = analyzer.plot_iv_chart(data, ticker, strike_price, expiration_date, option_type)
        if fig:
            plot_filename = f"{ticker}_{option_type}_{strike_price}_{expiration_date}_IV_chart.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {plot_filename}")
            plt.show()
    
    # Uncomment for multi-ticker comparison
    """
    print("\n" + "="*70)
    print("MULTI-TICKER COMPARISON")
    print("="*70)
    
    tickers = ["AAPL", "GOOGL", "MSFT"]
    fig_comp, all_data = analyzer.batch_analyze(
        tickers=tickers,
        expiration_date=expiration_date,
        strike=strike_price,
        option_type=option_type,
        days_back=days_back
    )
    
    if fig_comp:
        comp_filename = f"multi_ticker_IV_comparison_{expiration_date}.png"
        plt.savefig(comp_filename, dpi=300, bbox_inches='tight')
        print(f"\nComparison chart saved to: {comp_filename}")
        plt.show()
    """


if __name__ == "__main__":
    main()