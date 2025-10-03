"""
Historical Option Implied Volatility Plotter
Uses local Dolt database clone of post-no-preference/options
Queries via subprocess and plots IV over 90 days before expiration
Fetches both Call and Put data and displays them together
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
            return []
        
        # Parse JSON output
        data = json.loads(result.stdout)
        return data.get('rows', [])
    
    def find_nearest_expiration(self, ticker, target_expiration):
        """
        Find the nearest available expiration date
        """
        query = f"""
        SELECT DISTINCT expiration 
        FROM option_chain 
        WHERE act_symbol = '{ticker.upper()}'
        ORDER BY expiration
        """
        
        rows = self.query_dolt(query)
        
        if not rows:
            return None
        
        expirations = [row['expiration'] for row in rows]
        target_dt = datetime.strptime(target_expiration, '%Y-%m-%d')
        
        # Find nearest expiration
        nearest = min(expirations, key=lambda x: abs(
            datetime.strptime(x, '%Y-%m-%d') - target_dt
        ))
        
        return nearest
    
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
    
    def get_option_data(self, ticker, expiration_date, strike, days_back=90):
        """
        Fetch option data for BOTH call and put
        
        Parameters:
        -----------
        ticker: str - Stock ticker (e.g., 'AAPL')
        expiration_date: str - Expiration date in 'YYYY-MM-DD' format
        strike: float - Strike price
        days_back: int - Number of days before expiration (default 90)
        
        Returns:
        --------
        tuple: (call_data, put_data, actual_expiration, actual_strike)
        """
        # Calculate start date
        exp_dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        start_date = (exp_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # First, try exact match
        query_test = f"""
        SELECT date, strike, call_put, bid, ask, vol, delta, gamma, theta, vega
        FROM option_chain
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{expiration_date}'
        AND strike = {strike}
        AND date >= '{start_date}'
        AND date <= '{expiration_date}'
        ORDER BY date, call_put
        """
        
        test_rows = self.query_dolt(query_test)
        has_exact_match = bool(test_rows)
        
        actual_expiration = expiration_date
        actual_strike = strike
        
        if not has_exact_match:
            print(f"No exact match found for expiration={expiration_date}, strike={strike}")
            print("Searching for nearest available options...")
            
            # Find nearest expiration
            actual_expiration = self.find_nearest_expiration(ticker, expiration_date)
            if actual_expiration is None:
                raise ValueError(f"No options found for {ticker}")
            
            if actual_expiration != expiration_date:
                print(f"  Using nearest expiration: {actual_expiration} (requested: {expiration_date})")
                # Recalculate start date based on actual expiration
                exp_dt = datetime.strptime(actual_expiration, '%Y-%m-%d')
                start_date = (exp_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Find nearest strike
            actual_strike = self.find_nearest_strike(ticker, actual_expiration, strike)
            if actual_strike is None:
                raise ValueError(f"No strikes found for {ticker} on {actual_expiration}")
            
            if actual_strike != strike:
                print(f"  Using nearest strike: ${actual_strike:.2f} (requested: ${strike:.2f})")
        else:
            print(f"Found exact match: expiration={expiration_date}, strike=${strike:.2f}")
        
        # Query for BOTH call and put data
        query = f"""
        SELECT date, strike, call_put, bid, ask, vol, delta, gamma, theta, vega
        FROM option_chain
        WHERE act_symbol = '{ticker.upper()}'
        AND expiration = '{actual_expiration}'
        AND strike = {actual_strike}
        AND date >= '{start_date}'
        AND date <= '{actual_expiration}'
        ORDER BY date, call_put
        """
        
        print(f"Querying {ticker} Call and Put data from {start_date} to {actual_expiration}...")
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
        df['days_to_expiration'] = (pd.to_datetime(actual_expiration) - df['date']).dt.days
        
        # Split into call and put
        call_data = df[df['call_put'] == 'Call'].copy()
        put_data = df[df['call_put'] == 'Put'].copy()
        
        return call_data, put_data, actual_expiration, actual_strike
    
    def plot_main_chart(self, call_data, put_data, ticker, strike, expiration_date):
        """
        Create main visualization with IV, Option Price, and Vega
        Shows both Call and Put on same charts
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Colors for Call and Put
        call_color = '#2E86AB'
        put_color = '#D62246'
        
        # Plot 1: Implied Volatility
        ax1 = axes[0]
        
        if not call_data.empty:
            ax1.plot(call_data['date'], call_data['iv_percent'], 
                    linewidth=2.5, color=call_color, marker='o', markersize=6,
                    label='Call IV', alpha=0.8)
            ax1.fill_between(call_data['date'], call_data['iv_percent'], 
                            alpha=0.2, color=call_color)
            
            # Add average line for call
            avg_call_iv = call_data['iv_percent'].mean()
            ax1.axhline(y=avg_call_iv, color=call_color, linestyle='--', 
                       linewidth=1.5, alpha=0.5, label=f'Call Avg: {avg_call_iv:.2f}%')
        
        if not put_data.empty:
            ax1.plot(put_data['date'], put_data['iv_percent'], 
                    linewidth=2.5, color=put_color, marker='s', markersize=6,
                    label='Put IV', alpha=0.8)
            ax1.fill_between(put_data['date'], put_data['iv_percent'], 
                            alpha=0.2, color=put_color)
            
            # Add average line for put
            avg_put_iv = put_data['iv_percent'].mean()
            ax1.axhline(y=avg_put_iv, color=put_color, linestyle='--', 
                       linewidth=1.5, alpha=0.5, label=f'Put Avg: {avg_put_iv:.2f}%')
            
        # Get min and max
        min_iv = min(put_data['iv_percent'].min(), call_data['iv_percent'].min())
        max_iv = max(put_data['iv_percent'].max(), call_data['iv_percent'].max())

        # Expand range by ±5%
        ylim_lower = min_iv - 5
        ylim_upper = max_iv + 5

        # Apply to plot
        ax1.set_ylim(ylim_lower, ylim_upper)
        
        ax1.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
        ax1.set_title(
            f'{ticker} Option - Implied Volatility (90 Days)\n'
            f'Strike: ${strike:.2f}, Expiration: {expiration_date}',
            fontsize=14, fontweight='bold', pad=15
        )
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10, ncol=2)
        
        # Plot 2: Option Price
        ax2 = axes[1]
        
        if not call_data.empty:
            ax2.plot(call_data['date'], call_data['option_price'], 
                    linewidth=2.5, color=call_color, marker='o', markersize=5,
                    label='Call Price', alpha=0.8)
            ax2.fill_between(call_data['date'], call_data['option_price'], 
                            alpha=0.2, color=call_color)
        
        if not put_data.empty:
            ax2.plot(put_data['date'], put_data['option_price'], 
                    linewidth=2.5, color=put_color, marker='s', markersize=5,
                    label='Put Price', alpha=0.8)
            ax2.fill_between(put_data['date'], put_data['option_price'], 
                            alpha=0.2, color=put_color)
        
        ax2.set_ylabel('Option Price ($)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=10)
        
        # Plot 3: Vega
        ax3 = axes[2]
        
        if not call_data.empty and 'vega' in call_data.columns:
            ax3.plot(call_data['date'], call_data['vega'], 
                    linewidth=2.5, color=call_color, marker='o', markersize=5,
                    label='Call Vega', alpha=0.8)
            ax3.fill_between(call_data['date'], call_data['vega'], 
                            alpha=0.2, color=call_color)
        
        if not put_data.empty and 'vega' in put_data.columns:
            ax3.plot(put_data['date'], put_data['vega'], 
                    linewidth=2.5, color=put_color, marker='s', markersize=5,
                    label='Put Vega', alpha=0.8)
            ax3.fill_between(put_data['date'], put_data['vega'], 
                            alpha=0.2, color=put_color)
        
        ax3.set_ylabel('Vega', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(loc='best', fontsize=10)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def plot_greeks(self, call_data, put_data, ticker, strike, expiration_date):
        """
        Create detailed Greeks plot (optional, for --full flag)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        call_color = '#2E86AB'
        put_color = '#D62246'
        
        greeks = ['delta', 'gamma', 'theta', 'vega']
        
        for idx, greek in enumerate(greeks):
            ax = axes[idx // 2, idx % 2]
            
            # Plot call greek
            if not call_data.empty and greek in call_data.columns:
                ax.plot(call_data['date'], call_data[greek], 
                       linewidth=2.5, color=call_color, marker='o', markersize=5,
                       label=f'Call {greek.capitalize()}', alpha=0.8)
                ax.fill_between(call_data['date'], call_data[greek], 
                               alpha=0.2, color=call_color)
            
            # Plot put greek
            if not put_data.empty and greek in put_data.columns:
                ax.plot(put_data['date'], put_data[greek], 
                       linewidth=2.5, color=put_color, marker='s', markersize=5,
                       label=f'Put {greek.capitalize()}', alpha=0.8)
                ax.fill_between(put_data['date'], put_data[greek], 
                               alpha=0.2, color=put_color)
            
            ax.set_ylabel(greek.capitalize(), fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=10)
            ax.legend(loc='best', fontsize=9)
        
        fig.suptitle(
            f'{ticker} Option Greeks\n'
            f'Strike: ${strike:.2f}, Expiration: {expiration_date}',
            fontsize=14, fontweight='bold'
        )
        
        plt.tight_layout()
        
        return fig


def main():
    """
    Main function with user input
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot historical option IV')
    parser.add_argument('--full', action='store_true', 
                       help='Generate full Greeks chart in addition to main chart')
    args = parser.parse_args()
    
    print("="*70)
    print("Historical Option Implied Volatility Analyzer")
    print("Using local Dolt database: post-no-preference/options")
    print("="*70)
    print()
    
    # Get user input
    ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    expiration_date = input("Enter expiration date (YYYY-MM-DD): ").strip()
    strike = float(input("Enter strike price (e.g., 230.00): ").strip())
    
    # Optional: customize days back
    days_input = input("Days before expiration to analyze (default 90): ").strip()
    days_back = int(days_input) if days_input else 90
    
    print("\n" + "="*70)
    print("Fetching data...")
    print("="*70 + "\n")
    
    # Initialize plotter
    plotter = DoltOptionIVPlotter(dolt_repo_path="./options")
    
    try:
        # Get data for both call and put
        call_data, put_data, actual_expiration, actual_strike = plotter.get_option_data(
            ticker=ticker,
            expiration_date=expiration_date,
            strike=strike,
            days_back=days_back
        )
        
        # Print summary
        print(f"\n✓ Retrieved data:")
        if not call_data.empty:
            print(f"  Call: {len(call_data)} days")
            print(f"    Date range: {call_data['date'].min().date()} to {call_data['date'].max().date()}")
            print(f"    IV range: {call_data['iv_percent'].min():.2f}% to {call_data['iv_percent'].max():.2f}%")
            print(f"    Average IV: {call_data['iv_percent'].mean():.2f}%")
        
        if not put_data.empty:
            print(f"  Put: {len(put_data)} days")
            print(f"    Date range: {put_data['date'].min().date()} to {put_data['date'].max().date()}")
            print(f"    IV range: {put_data['iv_percent'].min():.2f}% to {put_data['iv_percent'].max():.2f}%")
            print(f"    Average IV: {put_data['iv_percent'].mean():.2f}%")
        
        # Combine data for CSV
        combined_data = pd.concat([call_data, put_data], ignore_index=True)
        
        # Save to CSV with organized directory structure
        csv_filename = f"./csv/{ticker}/{actual_expiration}/{actual_strike:.2f}_IV.csv"
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        combined_data.to_csv(csv_filename, index=False)
        print(f"\n✓ Data saved to: {csv_filename}")
        
        # Create main chart (IV, Price, Vega)
        print("\nGenerating main chart (IV, Price, Vega)...")
        fig_main = plotter.plot_main_chart(call_data, put_data, ticker, 
                                          actual_strike, actual_expiration)
        
        if fig_main:
            iv_filename = f"./graphs/{ticker}/{actual_expiration}/{actual_strike:.2f}_IV.png"
            os.makedirs(os.path.dirname(iv_filename), exist_ok=True)
            plt.figure(fig_main.number)
            plt.savefig(iv_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Main chart saved to: {iv_filename}")
        
        # Create Greeks chart if --full flag is used
        if args.full:
            print("\nGenerating full Greeks chart...")
            fig_greeks = plotter.plot_greeks(call_data, put_data, ticker, 
                                            actual_strike, actual_expiration)
            
            if fig_greeks:
                greeks_filename = f"./graphs/{ticker}/{actual_expiration}/{actual_strike:.2f}_Greeks.png"
                os.makedirs(os.path.dirname(greeks_filename), exist_ok=True)
                plt.figure(fig_greeks.number)
                plt.savefig(greeks_filename, dpi=300, bbox_inches='tight')
                print(f"✓ Greeks chart saved to: {greeks_filename}")
        
        print("\n" + "="*70)
        print("Analysis complete! Displaying charts...")
        if not args.full:
            print("(Use --full flag to also generate detailed Greeks chart)")
        print("="*70)
        
        plt.show()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()