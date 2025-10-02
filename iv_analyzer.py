"""
Historical Stock Option Implied Volatility Calculator
Uses IBKR API to fetch historical option data and calculate IV using reverse Black-Scholes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
from ib_insync import IB, Option, Stock
import asyncio
import warnings
warnings.filterwarnings('ignore')

class OptionIVCalculator:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """Initialize connection to IBKR"""
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        
    def connect(self):
        """Connect to IBKR TWS/Gateway"""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            print(f"Connected to IBKR on port {self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Make sure TWS/IB Gateway is running and API connections are enabled")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        self.ib.disconnect()
        print("Disconnected from IBKR")
    
    def black_scholes_price(self, S, K, T, r, sigma, option_type='call', q=0):
        """
        Calculate Black-Scholes option price
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility (IV we're solving for)
        option_type: 'call' or 'put'
        q: Dividend yield
        """
        if T <= 0:
            # Option has expired
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    
    def calculate_implied_volatility(self, option_price, S, K, T, r, option_type='call', q=0):
        """
        Calculate implied volatility using reverse Black-Scholes
        Returns IV as a decimal (e.g., 0.25 for 25%)
        """
        if T <= 0:
            return np.nan
        
        # Check for arbitrage violations
        if option_type == 'call':
            intrinsic = max(S - K, 0)
            if option_price < intrinsic:
                return np.nan
        else:
            intrinsic = max(K - S, 0)
            if option_price < intrinsic:
                return np.nan
        
        # Objective function to minimize
        def objective(sigma):
            if sigma <= 0:
                return 1e10
            try:
                bs_price = self.black_scholes_price(S, K, T, r, sigma, option_type, q)
                return abs(bs_price - option_price)
            except:
                return 1e10
        
        # Minimize to find IV
        result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
        
        if result.success and result.fun < 0.01:  # Good convergence
            return result.x
        else:
            return np.nan
    
    def get_contract_details(self, ticker, expiration_date, strike, option_type):
        """
        Find the nearest available option contract
        expiration_date: string in format 'YYYYMMDD'
        option_type: 'C' for call, 'P' for put
        """
        stock = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        
        # Get option chain for the expiration date
        chains = self.ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

        print(f"chains: {chains}")
        
        # Find the right chain
        target_exp = expiration_date
        available_strikes = []
        
        for chain in chains:
            if target_exp in chain.expirations:
                available_strikes = sorted(chain.strikes)
                break
        
        if not available_strikes:
            raise ValueError(f"No options found for {ticker} expiring on {expiration_date}")
        
        # Find nearest strike
        nearest_strike = min(available_strikes, key=lambda x: abs(x - strike))
        print(f"Requested strike: {strike}, Using nearest: {nearest_strike}")
        
        # Create option contract
        option = Option(ticker, expiration_date, nearest_strike, option_type, 'SMART')
        self.ib.qualifyContracts(option)
        
        return option, nearest_strike
    
    def get_historical_data(self, contract, end_date, days_back=90, bar_size='1 day'):
        """
        Fetch historical data from IBKR
        end_date: datetime object or string 'YYYYMMDD'
        """
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y%m%d')
        
        duration = f"{days_back} D"
        
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                df = pd.DataFrame(bars)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_stock_price_history(self, ticker, end_date, days_back=90):
        """Get historical stock prices"""
        stock = Stock(ticker, 'SMART', 'USD')
        self.ib.qualifyContracts(stock)
        
        return self.get_historical_data(stock, end_date, days_back)
    
    def get_treasury_rate(self, date):
        """
        Get approximate 3-month Treasury rate for a given date
        Uses a simplified approach - in production, fetch from FRED or similar
        """
        # For now, using rough approximations based on 2024-2025 rates
        # In production, you'd fetch actual historical rates from FRED API
        year = date.year
        
        if year >= 2024:
            return 0.045  # Approximate 4.5% for 2024-2025
        elif year >= 2023:
            return 0.05
        elif year >= 2022:
            return 0.03
        else:
            return 0.02
        
        # TODO: Implement actual FRED API fetch for production use
    
    def get_dividend_yield(self, ticker, date):
        """
        Estimate dividend yield at a given date
        Simplified approach - uses recent dividend data
        """
        try:
            stock = Stock(ticker, 'SMART', 'USD')
            self.ib.qualifyContracts(stock)
            
            # Get fundamental data
            fundamentals = self.ib.reqFundamentalData(stock, 'ReportSnapshot')
            
            # Parse dividend yield if available (simplified)
            # In production, you'd need historical dividend data
            return 0.01  # Default 1% if not available
        except:
            return 0.01  # Default assumption
    
    def calculate_iv_series(self, ticker, expiration_date, strike, option_type, days_back=90):
        """
        Main function to calculate IV over time
        expiration_date: string 'YYYYMMDD'
        option_type: 'call' or 'put' (will be converted to 'C' or 'P')
        """
        # Convert option type
        opt_type_code = 'C' if option_type.lower() == 'call' else 'P'
        
        print(f"\n{'='*60}")
        print(f"Analyzing {ticker} {option_type.upper()} option")
        print(f"Expiration: {expiration_date}, Target Strike: {strike}")
        print(f"{'='*60}\n")
        
        # Get option contract
        option, actual_strike = self.get_contract_details(ticker, expiration_date, strike, opt_type_code)
        
        # Get historical option prices
        print("Fetching historical option prices...")
        option_hist = self.get_historical_data(option, expiration_date, days_back)
        
        if option_hist.empty:
            print("No historical option data available")
            return None
        
        # Get historical stock prices
        print("Fetching historical stock prices...")
        stock_hist = self.get_historical_data(
            Stock(ticker, 'SMART', 'USD'), 
            expiration_date, 
            days_back
        )
        
        if stock_hist.empty:
            print("No historical stock data available")
            return None
        
        # Merge data
        stock_hist = stock_hist[['date', 'close']].rename(columns={'close': 'stock_price'})
        option_hist = option_hist[['date', 'close']].rename(columns={'close': 'option_price'})
        
        merged = pd.merge(option_hist, stock_hist, on='date', how='inner')
        
        if merged.empty:
            print("No overlapping data between option and stock prices")
            return None
        
        # Calculate IV for each day
        expiration_dt = datetime.strptime(expiration_date, '%Y%m%d')
        
        iv_data = []
        for _, row in merged.iterrows():
            date = row['date']
            days_to_exp = (expiration_dt - date).days
            T = days_to_exp / 365.0
            
            if T <= 0:
                continue
            
            r = self.get_treasury_rate(date)
            q = self.get_dividend_yield(ticker, date)
            
            iv = self.calculate_implied_volatility(
                row['option_price'],
                row['stock_price'],
                actual_strike,
                T,
                r,
                option_type.lower(),
                q
            )
            
            iv_data.append({
                'date': date,
                'days_to_expiration': days_to_exp,
                'stock_price': row['stock_price'],
                'option_price': row['option_price'],
                'implied_volatility': iv,
                'iv_percent': iv * 100 if not np.isnan(iv) else np.nan
            })
        
        result_df = pd.DataFrame(iv_data)
        result_df = result_df.dropna(subset=['implied_volatility'])
        
        print(f"\nCalculated IV for {len(result_df)} days")
        print(f"Average IV: {result_df['iv_percent'].mean():.2f}%")
        print(f"IV Range: {result_df['iv_percent'].min():.2f}% - {result_df['iv_percent'].max():.2f}%")
        
        return result_df
    
    def plot_iv_chart(self, iv_data, ticker, strike, expiration_date, option_type):
        """Create visualization of IV over time"""
        if iv_data is None or iv_data.empty:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot IV
        ax1.plot(iv_data['date'], iv_data['iv_percent'], 
                linewidth=2, color='blue', marker='o', markersize=4)
        ax1.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'{ticker} {option_type.upper()} Option Implied Volatility\n'
            f'Strike: ${strike}, Expiration: {expiration_date}',
            fontsize=14, fontweight='bold'
        )
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=iv_data['iv_percent'].mean(), color='r', 
                   linestyle='--', label=f'Average: {iv_data["iv_percent"].mean():.2f}%')
        ax1.legend()
        
        # Plot stock price
        ax2.plot(iv_data['date'], iv_data['stock_price'], 
                linewidth=2, color='green', marker='o', markersize=4)
        ax2.set_ylabel('Stock Price ($)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=strike, color='r', linestyle='--', 
                   label=f'Strike: ${strike}')
        ax2.legend()
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        return fig


def main():
    """
    Main execution function
    Example usage with configurable parameters
    """
    
    # CONFIGURATION
    tickers = ["AAPL", "GOOGL", "AMZN", "META", "MSFT", "NVDA", "TSLA"]
    
    # Choose one ticker for this run (or loop through all)
    ticker = "AAPL"
    
    # Option parameters
    strike_price = 150  # Target strike price
    expiration_date = "20250201"  # YYYYMMDD format
    option_type = "call"  # 'call' or 'put'
    
    # Analysis parameters
    days_back = 90  # How many days before expiration to analyze
    
    # IBKR connection settings
    host = '127.0.0.1'
    port = 7497  # 7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper, 4001 for Gateway live
    
    # Initialize calculator
    calc = OptionIVCalculator(host=host, port=port)
    
    # Connect to IBKR
    if not calc.connect():
        print("Failed to connect. Please check:")
        print("1. TWS or IB Gateway is running")
        print("2. API connections are enabled in TWS (File -> Global Configuration -> API -> Settings)")
        print("3. Port number is correct")
        return
    
    try:
        # Calculate IV series
        iv_data = calc.calculate_iv_series(
            ticker=ticker,
            expiration_date=expiration_date,
            strike=strike_price,
            option_type=option_type,
            days_back=days_back
        )
        
        if iv_data is not None:
            # Save to CSV
            filename = f"{ticker}_{option_type}_{strike_price}_{expiration_date}_IV.csv"
            iv_data.to_csv(filename, index=False)
            print(f"\nData saved to {filename}")
            
            # Create plot
            fig = calc.plot_iv_chart(iv_data, ticker, strike_price, 
                                    expiration_date, option_type)
            if fig:
                plot_filename = f"{ticker}_{option_type}_{strike_price}_{expiration_date}_IV.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Chart saved to {plot_filename}")
                plt.show()
        
    finally:
        # Always disconnect
        calc.disconnect()


if __name__ == "__main__":
    main()