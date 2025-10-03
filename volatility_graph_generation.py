"""
Historical Option Implied Volatility Plotter
Uses local Dolt database clone of post-no-preference/options
Queries via subprocess and plots IV over 90 days before expiration
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import os

class DoltOptionIVPlotter:
    def __init__(self, dolt_repo_path="./options"):
        """
        Initialize with path to local Dolt repository
        
        Parameters:
        -----------
        dolt_repo_path: str - Path to the cloned Dolt options repository
        """
        self.dolt_repo_path = dolt_repo_path
        
    def query_dolt(self, query):
        """
        Execute SQL query using Dolt CLI
        
        Returns:
        --------
        List of row dictionaries
        """
        result = subprocess.run(
            ["dolt", "sql", "-q", query, "-r", "json"],
            capture_output=True,
            text=True,
            cwd=self.dolt_repo_path
        )
        
        if result.returncode != 0:
            print("Dolt query failed:", result.stderr)
            raise RuntimeError("Query failed")
        
        if not result.stdout.strip():
            raise ValueError("No output from Dolt query")
        
        # Parse JSON output
        data = json.loads(result.stdout)
        return data.get('rows', [])
    
    def find_nearest_strike(self, ticker, expiration_date, target_strike):
        """
        Find the nearest available strike price
        """
        query = f"""
        SELECT DISTINCT strike 
        FROM option_chain 
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{expiration_date}'
        ORDER BY strike
        """
        
        rows = self.query_dolt(query)
        
        if not rows:
            return None
        
        strikes = [float(row['strike']) for row in rows]
        nearest_strike = min(strikes, key=lambda x: abs(x - target_strike))
        
        return nearest_strike
    
    def get_option_iv_data(self, ticker, expiration_date, strike, call_put, days_back=90):
        """
        Fetch option IV data for 90 days before expiration
        
        Parameters:
        -----------
        ticker: str - Stock ticker (e.g., 'AAPL')
        expiration_date: str - Expiration date in 'YYYY-MM-DD' format
        strike: float - Strike price
        call_put: str - 'Call' or 'Put'
        days_back: int - Number of days before expiration (default 90)
        
        Returns:
        --------
        pandas DataFrame with IV history
        """
        # Calculate start date
        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        start_date = (exp_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Normalize call_put to match database format
        call_put = call_put.capitalize()  # 'Call' or 'Put'
        
        print(f"Searching for nearest strike to ${strike}...")
        actual_strike = self.find_nearest_strike(ticker, expiration_date, strike)
        
        if actual_strike is None:
            raise ValueError(f"No options found for {ticker} expiring on {expiration_date}")
        
        print(f"Using strike: ${actual_strike:.2f}")
        
        # Query for option data
        query = f"""
        SELECT date, strike, call_put, bid, ask, vol, delta, gamma, theta, vega
        FROM option_chain
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{expiration_date}'
        AND strike = {actual_strike}
        AND call_put = '{call_put}'
        AND date >= '{start_date}'
        AND date <= '{expiration_date}'
        ORDER BY date
        """
        
        print(f"Querying {ticker} {call_put} option data from {start_date} to {expiration_date}...")
        rows = self.query_dolt(query)
        
        if not rows:
            raise ValueError("No data found for the specified option")
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Convert data types
        df['date'] = pd.to_datetime(df['date'])
        df['strike'] = pd.to_numeric(df['strike'])
        df['bid'] = pd.to_numeric(df['bid'])
        df['ask'] = pd.to_numeric(df['ask'])
        df['vol'] = pd.to_numeric(df['vol'])
        
        # Convert Greeks if present
        for col in ['delta', 'gamma', 'theta', 'vega']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        df['option_price'] = (df['bid'] + df['ask']) / 2
        df['iv_percent'] = df['vol'] * 100  # Convert to percentage
        df['days_to_expiration'] = (pd.to_datetime(expiration_date) - df['date']).dt.days
        
        return df, actual_strike
    
    def plot_iv(self, data, ticker, strike, expiration_date, call_put):
        """
        Create visualization of IV over time
        """
        if data is None or data.empty:
            print("No data to plot")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Implied Volatility
        ax1 = axes[0]
        ax1.plot(data['date'], data['iv_percent'], 
                linewidth=2.5, color='#2E86AB', marker='o', markersize=6,
                label='Implied Volatility')
        ax1.fill_between(data['date'], data['iv_percent'], alpha=0.3, color='#2E86AB')
        
        # Add average line
        avg_iv = data['iv_percent'].mean()
        ax1.axhline(y=avg_iv, color='#A23B72', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_iv:.2f}%')
        
        ax1.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'{ticker} {call_put.upper()} Option - Implied Volatility (90 Days)\n'
            f'Strike: ${strike:.2f}, Expiration: {expiration_date}',
            fontsize=14, fontweight='bold', pad=15
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=11)
        
        # Add statistics text box
        stats_text = (
            f'Min IV: {data["iv_percent"].min():.2f}%\n'
            f'Max IV: {data["iv_percent"].max():.2f}%\n'
            f'Std Dev: {data["iv_percent"].std():.2f}%'
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Plot 2: Option Price
        ax2 = axes[1]
        ax2.plot(data['date'], data['option_price'], 
                linewidth=2.5, color='#F18F01', marker='s', markersize=5,
                label='Mid Price')
        ax2.plot(data['date'], data['bid'], 
                linewidth=1.5, color='#06A77D', linestyle='--', alpha=0.7,
                label='Bid')
        ax2.plot(data['date'], data['ask'], 
                linewidth=1.5, color='#D62246', linestyle='--', alpha=0.7,
                label='Ask')
        ax2.fill_between(data['date'], data['bid'], data['ask'], 
                        alpha=0.2, color='#F18F01')
        
        ax2.set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=10)
        
        # Plot 3: Days to Expiration
        ax3 = axes[2]
        ax3.plot(data['date'], data['days_to_expiration'], 
                linewidth=2.5, color='#9C528B', marker='d', markersize=5)
        ax3.fill_between(data['date'], data['days_to_expiration'], 
                        alpha=0.3, color='#9C528B')
        
        ax3.set_ylabel('Days to Expiration', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.invert_yaxis()  # So time moves forward visually
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def plot_greeks(self, data, ticker, strike, expiration_date, call_put):
        """
        Create separate plot for Greeks
        """
        if data is None or data.empty:
            return None
        
        # Check if Greeks are available
        greek_cols = ['delta', 'gamma', 'theta', 'vega']
        available_greeks = [col for col in greek_cols if col in data.columns 
                           and not data[col].isna().all()]
        
        if not available_greeks:
            print("No Greeks data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = {'delta': '#06A77D', 'gamma': '#D62246', 
                 'theta': '#9C528B', 'vega': '#F4A261'}
        
        for idx, greek in enumerate(['delta', 'gamma', 'theta', 'vega']):
            ax = axes[idx // 2, idx % 2]
            
            if greek in available_greeks:
                ax.plot(data['date'], data[greek], 
                       linewidth=2.5, color=colors[greek], 
                       marker='o', markersize=5)
                ax.fill_between(data['date'], data[greek], 
                               alpha=0.3, color=colors[greek])
                ax.set_ylabel(greek.capitalize(), fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'{greek.capitalize()} not available',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, style='italic', color='gray')
            
            ax.set_xlabel('Date', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=10)
        
        fig.suptitle(
            f'{ticker} {call_put.upper()} Option Greeks\n'
            f'Strike: ${strike:.2f}, Expiration: {expiration_date}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        return fig


def main():
    """
    Main function with user input
    """
    print("="*70)
    print("Historical Option Implied Volatility Analyzer")
    print("Using local Dolt database: post-no-preference/options")
    print("="*70)
    print()
    
    # Get user input
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    expiration_date = input("Enter expiration date (YYYY-MM-DD): ").strip()
    strike = float(input("Enter strike price (e.g., 230.00): ").strip())
    call_put = input("Enter option type (call/put): ").strip().capitalize()
    
    # Validate option type
    if call_put not in ['Call', 'Put']:
        print("Invalid option type. Must be 'call' or 'put'")
        return
    
    # Optional: customize days back
    days_input = input("Days before expiration to analyze (default 90): ").strip()
    days_back = int(days_input) if days_input else 90
    
    print("\n" + "="*70)
    print("Fetching data...")
    print("="*70 + "\n")
    
    # Initialize plotter
    plotter = DoltOptionIVPlotter(dolt_repo_path="./options")
    
    try:
        # Get data
        data, actual_strike = plotter.get_option_iv_data(
            ticker=ticker,
            expiration_date=expiration_date,
            strike=strike,
            call_put=call_put,
            days_back=days_back
        )
        
        print(f"\n✓ Retrieved {len(data)} days of data")
        print(f"  Date range: {data['date'].min().date()} to {data['date'].max().date()}")
        print(f"  IV range: {data['iv_percent'].min():.2f}% to {data['iv_percent'].max():.2f}%")
        print(f"  Average IV: {data['iv_percent'].mean():.2f}%")
        
        # Save to CSV
        csv_filename = f"./csv/{ticker}/{expiration_date}/{actual_strike:.2f}_{call_put}_IV.csv"
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        data.to_csv(csv_filename, index=False)
        print(f"\n✓ Data saved to: {csv_filename}")
        
        # Create IV plot
        print("\nGenerating plots...")
        fig_iv = plotter.plot_iv(data, ticker, actual_strike, expiration_date, call_put)
        
        if fig_iv:
            iv_filename = f"./graphs/{ticker}/{expiration_date}/{actual_strike:.2f}_{call_put}_IV.png"
            os.makedirs(os.path.dirname(iv_filename), exist_ok=True)
            plt.figure(fig_iv.number)
            plt.savefig(iv_filename, dpi=300, bbox_inches='tight')
            print(f"✓ IV chart saved to: {iv_filename}")
        
        # Create Greeks plot
        fig_greeks = plotter.plot_greeks(data, ticker, actual_strike, expiration_date, call_put)
        
        if fig_greeks:
            greeks_filename = f"./graphs/{ticker}/{expiration_date}/{actual_strike:.2f}_{call_put}_Greeks.png"
            os.makedirs(os.path.dirname(greeks_filename), exist_ok=True)
            plt.figure(fig_greeks.number)
            plt.savefig(greeks_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Greeks chart saved to: {greeks_filename}")
        
        print("\n" + "="*70)
        print("Analysis complete! Displaying charts...")
        print("="*70)
        
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()