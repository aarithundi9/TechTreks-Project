#!/usr/bin/env python3
"""
ETF Historical Data Fetcher using yfinance

This script fetches historical price data for ETFs using yfinance library.
It's designed to work with the Reddit sentiment analysis data for ML model development.

Author: TechTreks Project
Date: 2025-01-27
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ETFDataFetcher:
    """Class to fetch and process ETF historical data"""
    
    def __init__(self):
        """Initialize the ETF data fetcher"""
        self.data = {}
        self.etf_symbols = ['QQQ', 'DIA', 'IWM', 'EFA', 'VTI']
        
    def fetch_etf_data(self, symbol, period='2y', interval='1d'):
        """
        Fetch historical data for a specific ETF
        
        Args:
            symbol (str): ETF symbol (e.g., 'QQQ')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pandas.DataFrame: Historical data with OHLCV columns
        """
        try:
            print(f"Fetching {symbol} data for period: {period}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            hist_data = ticker.history(period=period, interval=interval)
            
            if hist_data.empty:
                print(f"⚠️  No data found for {symbol}")
                return None
            
            # Add some basic technical indicators
            hist_data = self._add_technical_indicators(hist_data)
            
            # Store the data
            self.data[symbol] = hist_data
            
            print(f"✓ Successfully fetched {len(hist_data)} records for {symbol}")
            print(f"  Date range: {hist_data.index[0].strftime('%Y-%m-%d')} to {hist_data.index[-1].strftime('%Y-%m-%d')}")
            
            return hist_data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_etfs(self, symbols=None, period='2y', interval='1d'):
        """
        Fetch data for multiple ETFs
        
        Args:
            symbols (list): List of ETF symbols
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            dict: Dictionary with symbol as key and DataFrame as value
        """
        if symbols is None:
            symbols = self.etf_symbols
        
        print(f"Fetching data for {len(symbols)} ETFs: {', '.join(symbols)}")
        print("=" * 60)
        
        successful_fetches = {}
        
        for symbol in symbols:
            data = self.fetch_etf_data(symbol, period, interval)
            if data is not None:
                successful_fetches[symbol] = data
            print("-" * 40)
        
        print(f"\n🎉 Successfully fetched data for {len(successful_fetches)} ETFs")
        return successful_fetches
    
    def _add_technical_indicators(self, df):
        """Add basic technical indicators to the data"""
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility (rolling standard deviation of returns)
        df['Volatility_5d'] = df['Daily_Return'].rolling(window=5).std()
        df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
        
        # Price momentum
        df['Price_Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Price_Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume indicators
        df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
        
        # Price position within recent range
        df['High_20d'] = df['High'].rolling(window=20).max()
        df['Low_20d'] = df['Low'].rolling(window=20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20d']) / (df['High_20d'] - df['Low_20d'])
        
        return df
    
    def plot_price_data(self, symbol, start_date=None, end_date=None):
        """Plot price data for a specific ETF"""
        if symbol not in self.data:
            print(f"❌ No data available for {symbol}")
            return
        
        df = self.data[symbol].copy()
        
        # Filter by date range if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{symbol} Historical Data Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=2)
        axes[0].plot(df.index, df['MA_20'], label='20-day MA', alpha=0.7)
        axes[0].plot(df.index, df['MA_50'], label='50-day MA', alpha=0.7)
        axes[0].set_title('Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Volume
        axes[1].bar(df.index, df['Volume'], alpha=0.6, color='orange')
        axes[1].plot(df.index, df['Volume_MA_10'], color='red', linewidth=2, label='10-day Volume MA')
        axes[1].set_title('Trading Volume')
        axes[1].set_ylabel('Volume')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Daily Returns and Volatility
        axes[2].plot(df.index, df['Daily_Return'], alpha=0.7, label='Daily Returns')
        axes[2].plot(df.index, df['Volatility_20d'], color='red', linewidth=2, label='20-day Volatility')
        axes[2].set_title('Daily Returns and Volatility')
        axes[2].set_ylabel('Return / Volatility')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n=== {symbol} Summary Statistics ===")
        print(f"Total trading days: {len(df)}")
        print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
        print(f"Average daily return: {df['Daily_Return'].mean():.4f}")
        print(f"Daily return volatility: {df['Daily_Return'].std():.4f}")
        print(f"Total return: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.2f}%")
    
    def save_data(self, symbol, filename=None):
        """Save ETF data to CSV file"""
        if symbol not in self.data:
            print(f"❌ No data available for {symbol}")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_historical_data_{timestamp}.csv"
        
        self.data[symbol].to_csv(filename)
        print(f"✓ Saved {symbol} data to {filename}")
        print(f"  Shape: {self.data[symbol].shape}")
        print(f"  Columns: {list(self.data[symbol].columns)}")
    
    def get_data_summary(self):
        """Get summary of all fetched data"""
        if not self.data:
            print("❌ No data available")
            return
        
        print("\n=== ETF Data Summary ===")
        for symbol, df in self.data.items():
            print(f"\n{symbol}:")
            print(f"  Records: {len(df)}")
            print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"  Current price: ${df['Close'].iloc[-1]:.2f}")
            print(f"  Avg daily return: {df['Daily_Return'].mean():.4f}")
            print(f"  Volatility: {df['Daily_Return'].std():.4f}")


def main():
    """Main function to demonstrate usage"""
    print("🚀 ETF Historical Data Fetcher")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = ETFDataFetcher()
    
    # Fetch QQQ data (as requested)
    print("\n📊 Fetching QQQ data...")
    qqq_data = fetcher.fetch_etf_data('QQQ', period='2y')
    
    if qqq_data is not None:
        # Plot the data
        fetcher.plot_price_data('QQQ')
        
        # Save the data
        fetcher.save_data('QQQ')
        
        # Show summary
        fetcher.get_data_summary()
        
        # Show sample data
        print("\n=== Sample QQQ Data ===")
        print(qqq_data.head())
        print("\n=== Data Info ===")
        print(qqq_data.info())
    
    # Optional: Fetch all ETFs
    print("\n" + "="*60)
    print("🔄 Would you like to fetch data for all ETFs? (QQQ, DIA, IWM, EFA, VTI)")
    print("Uncomment the following lines to fetch all ETFs:")
    print("# all_data = fetcher.fetch_multiple_etfs()")
    print("# for symbol in all_data.keys():")
    print("#     fetcher.save_data(symbol)")


if __name__ == "__main__":
    main()
