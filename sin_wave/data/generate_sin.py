# data_generation/synthetic_data.py

import numpy as np
import pandas as pd
import ta

def generate_sine_wave_data(
    n_cycles=10,
    points_per_cycle=360,
    noise_factor=0.05,
    seed=None
):
    """
    Generates synthetic sine wave data and calculates 
    Open, High, Low, Close, RSI, EMA, MACD, etc.

    Parameters
    ----------
    n_cycles : int
        Number of sine wave cycles.
    points_per_cycle : int
        Points per cycle.
    noise_factor : float
        How much random noise to add.
    seed : int or None
        Random seed, for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing OHLC data and technical indicators.
    """

    if seed is not None:
        np.random.seed(seed)
        
    # Generate time steps
    time = np.arange(0, n_cycles * 2 * np.pi, 2 * np.pi / points_per_cycle)
    
    # Base sine wave
    base_price = np.sin(time)
    
    # Add noise
    noise = noise_factor * np.random.randn(len(base_price))  
    synthetic_price = base_price + noise
    
    # Create synthetic Open/High/Low
    open_price = synthetic_price + 0.01 * np.random.randn(len(synthetic_price))
    high_price = np.maximum(synthetic_price, open_price) + 0.02
    low_price  = np.minimum(synthetic_price, open_price) - 0.02
    
    # Build DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': synthetic_price
    })
    
    # Add technical indicators
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    df['ema_fast'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['ema_slow'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['macd'] = df['ema_fast'] - df['ema_slow']
    
    # Drop NaNs and reset index if desired
    df = df.iloc[26:].reset_index(drop=True)
    
    return df


def main():
    """Quick test to show usage of generate_sine_wave_data."""
    df = generate_sine_wave_data()
    print(df.head())

if __name__ == "__main__":
    main()

