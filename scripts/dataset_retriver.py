from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import h5py
from tqdm import tqdm


class data_retriever:

    def __init__(self,key_filename = 'keys.txt'):
        '''
        Initilises a wrapper with the binance api client using the keys

        Returns
        -------
        None.

        '''

        #reads keys from file
        with open(key_filename) as key_file:
            key = key_file.readline().strip('\n')
            secret = key_file.readline().strip('\n')
        
        self.__api_key = key
        self.__api_secret = secret
        
        #creates a binance client object for data collection
        self.client = Client(self.__api_key, self.__api_secret)

    def get_time_pages(self,start_time, end_time, interval, page_size):
        '''
        Divides the time range into pages and prints the page steps.

        Parameters
        ----------
        start_time : int
            The start time in UNIX timestamp.
        end_time : int
            The end time in UNIX timestamp.
        interval : str
            The time interval (e.g., '1m' for 1 minute, '1h' for 1 hour).
        page_size : int
            The number of intervals per page.
        
        Prints each time page in the format (start, end).
        '''
        
        # Define the interval length in seconds based on the interval string
        interval_seconds = {
            '1s': 1,
            '1m': 60,
            '5m': 5 * 60,
            '15m': 15 * 60,
            '30m': 30 * 60,
            '1h': 60 * 60,
            '4h': 4 * 60 * 60,
            '1d': 24 * 60 * 60
        }

        # Ensure the interval is valid
        if interval not in interval_seconds:
            raise ValueError(f"Invalid interval. Valid options are: {', '.join(interval_seconds.keys())}")

        # Get the number of seconds per interval
        interval_length = interval_seconds[interval]

        # Calculate the number of intervals per page (page_size)
        page_step = page_size * interval_length

        steps = []
        # Loop over the total range and print the pages
        current_time = start_time
        while current_time < end_time:
            next_time = current_time + page_step
            if next_time > end_time:  # If next_time exceeds end_time, set it to end_time
                next_time = end_time
            steps.append((current_time,next_time))
            current_time = next_time
        
        return steps
    
    def get_time_range(self,amount, interval):
        '''
        Given an amount and interval, returns the current Unix time and the Unix time
        adjusted by the amount * interval.

        Parameters
        ----------
        amount : int
            The amount of time units to subtract.
        interval : int
            The time interval in units of minutes, seconds, etc.

        Returns
        -------
        tuple
            A tuple containing the current Unix timestamp and the calculated adjusted Unix timestamp.
        '''

        interval_seconds = {
            '1s': 1,
            '1m': 60,
            '5m': 5 * 60,
            '15m': 15 * 60,
            '30m': 30 * 60,
            '1h': 60 * 60,
            '4h': 4 * 60 * 60,
            '1d': 24 * 60 * 60
        }

        # Get the current time in Unix timestamp
        now = int(datetime.now(timezone.utc).timestamp())
        
        # Calculate the adjusted timestamp by subtracting the amount * interval
        # Subtract based on the unit of time (e.g., seconds, minutes)
        adjusted_time = now - (amount * interval_seconds.get(interval,60))

        return now, adjusted_time
            
    def get_price_page(self, symbol, start_time, end_time, interval='1m'):
        '''
        Parameters
        ----------
        symbol : str
            The required token price (e.g., BTCUSDT).
        start_time : str or int
            The start time for fetching data, either as a UNIX timestamp or a date string.
        end_time : str or int
            The end time for fetching data, either as a UNIX timestamp or a date string.
        interval : str
            The Kline interval for price data. Options: '1m', '5m', '15m', '30m', '1h', '4h', '1d', etc.
            Default is '1m' (minute).
        return_data : bool
            Whether to return the data after fetching. Default is False.

        Returns
        -------
        klines : dict
            A dictionary of the historical price data for the specified symbol.
            e.g., klines['BTCUSDT']['open'] is a 1d numpy array of open prices over the interval.
        '''
        # Validate inputs
        assert isinstance(symbol, str), "symbol must be a string"
        assert isinstance(start_time, (str, int)), "start_time must be a string or integer"
        assert isinstance(end_time, (str, int)), "end_time must be a string or integer"
        assert isinstance(interval, str), "interval must be a string"

        # Valid Kline intervals (you can expand this as necessary)
        valid_intervals = ['1s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        assert interval in valid_intervals, f"Invalid interval. Choose from {valid_intervals}"

        # Convert start_time and end_time to UNIX timestamps if they are in string format
        if isinstance(start_time, str):
            start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp())  # Convert to UNIX timestamp
        if isinstance(end_time, str):
            end_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp())  # Convert to UNIX timestamp

        # Convert start_time and end_time to datetime objects in UTC
        start_time_dt = datetime.fromtimestamp(start_time, timezone.utc)
        end_time_dt = datetime.fromtimestamp(end_time, timezone.utc)
        
        labeled_price_data = {}

        interval_dict = {
            '1s': Client.KLINE_INTERVAL_1SECOND,
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }

        INTERVAL_STEP = interval_dict[interval]

        # Make a single API call to fetch the data between start_time and end_time
        price_data = np.array(self.client.get_historical_klines(
            symbol, INTERVAL_STEP, str(start_time_dt), str(end_time_dt)
        ))
        
        # If no data is returned, print a message and return None
        if len(price_data) == 0:
            return None

        # Transpose and convert to float
        price_data = price_data.T
        price_data = price_data.astype(float)

        # Label the price data
        labeled_price_data['time'] = np.array(price_data[0])
        labeled_price_data['open'] = np.array(price_data[1])
        labeled_price_data['high'] = np.array(price_data[2])
        labeled_price_data['low'] = np.array(price_data[3])
        labeled_price_data['close'] = np.array(price_data[4])
        labeled_price_data['volume'] = np.array(price_data[5])

        return labeled_price_data

    def fetch_and_save_dataset(self, symbols, amount, interval,dir = "data/"):
        PAGE_SIZE = 1000
        filename = f"dataset_{amount}_{interval}.h5"  # Fixed extension to ".h5"
        filename = dir + filename 

        # Get time range and pages
        end_time, start_time = self.get_time_range(amount, interval)
        time_pages = self.get_time_pages(start_time, end_time, interval, PAGE_SIZE)

        for symbol in symbols:

            print(f"Collecting {interval} for {symbol} between {datetime.fromtimestamp(start_time)} and {datetime.fromtimestamp(end_time)}")
            # Open HDF5 file in append mode
            with h5py.File(filename, "a") as h5f:
                # Ensure symbol group exists
                if symbol not in h5f:
                    symbol_group = h5f.create_group(symbol)
                else:
                    del h5f[symbol]
                    print(f"Deleted existing data for {symbol}")
                    symbol_group = h5f.create_group(symbol)

                for page_start_time, page_end_time in tqdm(time_pages, desc=f"Fetching {symbol} {interval} prices", unit="page"):

                    page = self.get_price_page(symbol, page_start_time, page_end_time, interval)

                    if page is None or len(page) == 0:
                        print(f"Data collection for {symbol} failed.")
                        print(f"No data available between {datetime.fromtimestamp(start_time)} and {datetime.fromtimestamp(end_time)}.")

                        # Delete dataset if fetching fails
                        if symbol in h5f:
                            del h5f[symbol]
                            print(f"Deleted incomplete dataset for {symbol}.")
                        break

                    # Process each key in the dictionary separately
                    for key, values in page.items():
                        values_np = np.array(values, dtype="float64")  
                        
                        if key not in symbol_group:
                            # Create dataset for this key
                            symbol_group.create_dataset(
                                key, shape=(0,), maxshape=(None,), dtype="float64", compression="gzip"
                            )
                        
                        # Resize dataset and append
                        dataset = symbol_group[key]
                        current_size = dataset.shape[0]
                        new_size = current_size + values_np.shape[0]
                        dataset.resize(new_size, axis=0)
                        dataset[current_size:new_size] = values_np

            print(f"Dataset for {symbol} saved successfully in {filename}. \n")


if __name__ == "__main__":
    downloader = data_retriever()
    tokens = ['ETHUSDT','XRPUSDT','BTCUSDT']
    downloader.fetch_and_save_dataset(tokens,10000,'4h')
    downloader.fetch_and_save_dataset(tokens,60000,'1h')
    downloader.fetch_and_save_dataset(tokens,100000,'15m')
    downloader.fetch_and_save_dataset(tokens,100000,'5m')
    downloader.fetch_and_save_dataset(tokens,100000,'1m')

