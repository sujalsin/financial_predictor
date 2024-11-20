import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, symbols, start_date=None, end_date=None):
        """
        Initialize the DataCollector with stock symbols and date range.
        
        Args:
            symbols (list): List of stock symbols to collect data for
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_stock_data(self, symbol):
        """
        Fetch historical stock data for a given symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            stock = yf.Ticker(symbol)
            df = stock.history(start=self.start_date, end=self.end_date)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def collect_and_save(self):
        """
        Collect data for all symbols and save to CSV files.
        """
        for symbol in self.symbols:
            df = self.fetch_stock_data(symbol)
            if df is not None and not df.empty:
                output_file = self.data_dir / f"{symbol}_data.csv"
                df.to_csv(output_file)
                logger.info(f"Saved data for {symbol} to {output_file}")
            else:
                logger.warning(f"No data collected for {symbol}")

def main():
    # Example usage
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Example symbols
    collector = DataCollector(symbols)
    collector.collect_and_save()

if __name__ == "__main__":
    main()
