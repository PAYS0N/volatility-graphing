import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import options
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date

# --- 1. USER INPUTS (Modify these values) ---
TICKER = 'AAPL'
EXPIRATION_DATE = '2025-01-17'
STRIKE_PRICE = 170
OPTION_TYPE = 'call' # Use 'call' or 'put'
SMILE_DATE = '2025-01-10' # A date before expiration for the smile graph

# --- 2. Black-Scholes Implementation ---

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the Black-Scholes option price.
    S: Underlying asset price
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate
    sigma: Volatility (implied)
    """
    if T <= 0:
        # If expiration is today or in the past, return intrinsic value
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        
    return price

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculates the implied volatility using the Brentq root-finding method.
    """
    # Objective function: difference between market price and BS model price
    objective_function = lambda sigma: black_scholes(S, K, T, r, sigma, option_type) - market_price

    # Brentq needs a bracket [a, b] where f(a) and f(b) have opposite signs.
    # We search for a solution between 0.01% and 500% volatility.
    try:
        iv = brentq(objective_function, a=1e-4, b=5.0)
    except ValueError:
        iv = np.nan # Return NaN if no solution is found
        
    return iv

# --- 3. Data Fetching and Analysis ---

def analyze_historical_iv(ticker, expiration_date, strike, option_type):
    """
    Fetches historical data and calculates IV for a single option contract.
    """
    print(f"Fetching data for {ticker} {strike} {option_type.capitalize()} expiring {expiration_date}...")
    
    # Get historical stock data
    stock_data = yf.download(ticker, start="2020-01-01", end=expiration_date, progress=False)['Adj Close']
    
    # Get risk-free rate (13-week Treasury Bill)
    rf_data = yf.download('^IRX', start="2020-01-01", end=expiration_date, progress=False)['Adj Close'] / 100
    
    # Get historical option data
    try:
        hist_options = options.get_historical_option_data(ticker, expiration_date)
        if option_type == 'call':
            contract_data = hist_options['calls']
        else:
            contract_data = hist_options['puts']
        
        # Filter for the specific strike price
        contract_data = contract_data[contract_data['Strike'] == strike].copy()
        if contract_data.empty:
            print(f"Error: No historical data found for strike {strike}.")
            return None
            
    except Exception as e:
        print(f"Error fetching historical options data: {e}")
        return None

    # Prepare data for calculation
    contract_data['Date'] = pd.to_datetime(contract_data['Date'])
    contract_data.set_index('Date', inplace=True)
    
    # Combine all data sources
    df = pd.DataFrame(index=contract_data.index)
    df['Option_Price'] = (contract_data['Bid'] + contract_data['Ask']) / 2 # Use midpoint price
    df['Underlying_Price'] = stock_data
    df['Risk_Free_Rate'] = rf_data
    
    # Forward fill missing stock prices and risk-free rates (for weekends/holidays)
    df.ffill(inplace=True)
    df.dropna(inplace=True) # Drop any remaining rows with missing data
    
    exp_date_obj = datetime.strptime(expiration_date, '%Y-%m-%d').date()
    
    # Calculate IV for each day
    df['Time_to_Exp'] = [(exp_date_obj - idx.date()).days / 365.25 for idx in df.index]
    
    df['IV'] = np.vectorize(implied_volatility)(
        df['Option_Price'],
        df['Underlying_Price'],
        strike,
        df['Time_to_Exp'],
        df['Risk_Free_Rate'],
        option_type
    )
    
    return df

def analyze_volatility_smile(ticker, expiration_date, smile_date, option_type):
    """
    Fetches option chain for a single day to calculate the volatility smile.
    """
    print(f"\nFetching volatility smile data for {smile_date}...")
    try:
        chain = options.get_options_chain(ticker, date=smile_date)
        if option_type == 'call':
            df = chain['calls']
        else:
            df = chain['puts']

        # Get data needed for IV calculation
        stock_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[0]
        rf_rate = yf.download('^IRX', start=smile_date, progress=False)['Close'].iloc[0] / 100
        
        exp_date_obj = datetime.strptime(expiration_date, '%Y-%m-%d').date()
        smile_date_obj = datetime.strptime(smile_date, '%Y-%m-%d').date()
        time_to_exp = (exp_date_obj - smile_date_obj).days / 365.25

        # Calculate IV for each strike
        df['IV'] = df.apply(
            lambda row: implied_volatility(
                row['Last Price'], stock_price, row['Strike'], time_to_exp, rf_rate, option_type
            ), axis=1
        )
        # Filter out NaN values and unrealistic IVs
        df.dropna(subset=['IV'], inplace=True)
        df = df[df['IV'] > 0.01]
        return df

    except Exception as e:
        print(f"Could not fetch volatility smile data: {e}")
        return None

# --- 4. Main Execution and Plotting ---

if __name__ == "__main__":
    # --- Plot 1: Historical Implied Volatility ---
    hist_df = analyze_historical_iv(TICKER, EXPIRATION_DATE, STRIKE_PRICE, OPTION_TYPE)
    
    if hist_df is not None and not hist_df.empty:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot IV
        color = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Implied Volatility (%)', color=color)
        ax1.plot(hist_df.index, hist_df['IV'] * 100, color=color, label='Implied Volatility')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f'Historical Implied Volatility vs. Stock Price\n{TICKER} ${STRIKE_PRICE} {OPTION_TYPE.capitalize()} expiring {EXPIRATION_DATE}')
        
        # Plot Stock Price on a second y-axis
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Stock Price ($)', color=color)
        ax2.plot(hist_df.index, hist_df['Underlying_Price'], color=color, label='Stock Price')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
    # --- Plot 2: Volatility Smile ---
    smile_df = analyze_volatility_smile(TICKER, EXPIRATION_DATE, SMILE_DATE, OPTION_TYPE)

    if smile_df is not None and not smile_df.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(smile_df['Strike'], smile_df['IV'] * 100, marker='o', linestyle='-')
        plt.title(f'Volatility Smile for {TICKER} {OPTION_TYPE.capitalize()}s expiring {EXPIRATION_DATE}\n(Snapshot on {SMILE_DATE})')
        plt.xlabel('Strike Price ($)')
        plt.ylabel('Implied Volatility (%)')
        plt.grid(True)

    # Show all plots
    if (hist_df is not None and not hist_df.empty) or \
       (smile_df is not None and not smile_df.empty):
        plt.show()
    else:
        print("\nCould not generate any plots due to data fetching errors.")