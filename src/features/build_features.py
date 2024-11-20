import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        """
        Initialize the FeatureEngineer with paths to data directories.
        """
        self.data_dir = Path(__file__).parent.parent.parent / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()

    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal_line,
            'MACD_Hist': macd - signal_line
        })

    def calculate_bollinger_bands(self, data, window=20):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return pd.DataFrame({
            'BB_Middle': sma,
            'BB_Upper': upper_band,
            'BB_Lower': lower_band
        })

    def calculate_obv(self, data, volume):
        """Calculate On Balance Volume"""
        obv = np.where(data > data.shift(1), volume, -volume).cumsum()
        return obv

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators for the given dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        # Moving averages
        df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        df['SMA_50'] = self.calculate_sma(df['Close'], 50)
        df['EMA_20'] = self.calculate_ema(df['Close'], 20)
        
        # Momentum indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        macd_data = self.calculate_macd(df['Close'])
        df = pd.concat([df, macd_data], axis=1)
        
        # Volatility indicators
        bollinger = self.calculate_bollinger_bands(df['Close'])
        df = pd.concat([df, bollinger], axis=1)
        
        # Volume indicators
        df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
        
        return df

    def add_price_derived_features(self, df):
        """
        Add price-derived features to the dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        # Returns
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close']).diff()
        
        # Price ranges
        df['Daily_Range'] = df['High'] - df['Low']
        df['Daily_Range_Pct'] = df['Daily_Range'] / df['Close']
        
        # Gap features
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = df['Gap'] / df['Close'].shift(1)
        
        return df

    def create_target_variable(self, df, forecast_horizon=5):
        """
        Create target variable for prediction.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            forecast_horizon (int): Number of days to forecast ahead
            
        Returns:
            pd.DataFrame: DataFrame with target variable
        """
        df['Target_Return'] = df['Close'].pct_change(periods=forecast_horizon).shift(-forecast_horizon)
        df['Target'] = (df['Target_Return'] > 0).astype(int)  # 1 if price goes up, 0 if down
        return df

    def process_stock_data(self, symbol):
        """
        Process data for a single stock symbol.
        
        Args:
            symbol (str): Stock symbol
        """
        try:
            # Read raw data
            input_file = self.raw_dir / f"{symbol}_data.csv"
            df = pd.read_csv(input_file, index_col=0, parse_dates=True)
            
            # Calculate features
            df = self.calculate_technical_indicators(df)
            df = self.add_price_derived_features(df)
            df = self.create_target_variable(df)
            
            # Handle missing values
            df = df.dropna()
            
            # Save processed data
            output_file = self.processed_dir / f"{symbol}_processed.csv"
            df.to_csv(output_file)
            logger.info(f"Processed data saved for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing data for {symbol}: {str(e)}")

def main():
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    engineer = FeatureEngineer()
    
    for symbol in symbols:
        engineer.process_stock_data(symbol)

if __name__ == "__main__":
    main()
