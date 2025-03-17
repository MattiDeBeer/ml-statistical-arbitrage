'''
Use to determine suitable series. 
SPecify min volume and start timestamp to detmerine what pairs fit this criteria. otuputs can be used in dataset_retreiver script
'''

from binance.client import Client

client = Client()

# Define criteria
min_volume = 1_000_000  # Only include tokens with 24h trading volume above 10M USDT
start_timestamp = int(1546300800000.)  # 2019-01-01 00:00:00 UTC in milliseconds

# Get all active USDT pairs
exchange_info = client.get_exchange_info()
active_usdt_pairs = {
    s['symbol'] for s in exchange_info['symbols']
    if s['symbol'].endswith('USDT') and s['status'] == 'TRADING' and s['isSpotTradingAllowed']
}

# Get 24-hour ticker stats
tickers = client.get_ticker()

# Filter based on volume
high_volume_pairs = [
    ticker['symbol'] for ticker in tickers
    if ticker['symbol'] in active_usdt_pairs and float(ticker['quoteVolume']) > min_volume
]

# Check historical data availability
valid_pairs = []
for pair in high_volume_pairs:
    try:
        # Fetch first available Kline data
        first_candle = client.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_1DAY, startTime=0, limit=1)
        if first_candle and int(first_candle[0][0]) <= start_timestamp:
            valid_pairs.append(pair)  # Only add pairs that have data since 2018-05-13
    except Exception as e:
        print(f"Skipping {pair} due to error: {e}")

print(valid_pairs)  # USDT pairs with high volume that have data since 2018

