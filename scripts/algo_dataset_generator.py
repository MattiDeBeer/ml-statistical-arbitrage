import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm


def sample_gp(number,l1=0.5,l2=0.02,ratio = 0.1):
        x = np.linspace(0,1,number)
        
        x1 = np.outer(x,np.ones(number))
        x2 = np.outer(np.ones(number),x)

        cov1 = np.exp(-((x1-x2)**2)/l1**2)
        cov2 = np.exp(-((x1-x2)**2)/l2**2)

        samples1 = np.random.multivariate_normal(mean=np.zeros(number), cov=cov1, size=1)[0]
        samples2 = np.random.multivariate_normal(mean=np.zeros(number), cov=cov2, size=1)[0]

        return samples1 + ratio*samples2

def sample_OUP(T):
     # Parameters
        m = 0  # Long-term mean
        k = 1e3 / T  # Speed of reversion
        sigma = 0.1  # Volatility
        X0 = 0  # Initial value
        dt = 0.01  # Time step
        T = int(T*dt)

        # Simulation
        t = np.arange(0, T, dt)
        X = np.zeros_like(t)
        X[0] = X0

        for i in range(1, len(t)):
            Z = np.random.normal(0, 1)  # Standard normal random variable
            X[i] = X[i-1] + k*(m - X[i-1])*dt + sigma*np.sqrt(dt)*Z
        
        return X


def generate_trending_cointegrated_data(tokens, num_periods=1000, seed=None, noise=0.02,reversion_scale = 2):
    """
    Generates trending, cointegrated OHLC trading data for specified tokens.

    Args:
        tokens (list): List of token names (e.g., ["BTC", "ETH", "SOL"]).
        num_periods (int): Number of time periods to generate data for.
        seed (int, optional): Random seed for reproducibility.
        drift (float): Upward/downward drift component for trending prices.
        noise (float): Random noise to simulate market fluctuations.

    Returns:
        dict: Dictionary of NumPy arrays containing OHLC data for each token.
    """

    if seed:
        np.random.seed(seed)

    g_sample = sample_gp(num_periods,l1=0.5,l2=0.1,ratio=0.2)
    base_trend = g_sample + np.min(g_sample) + 1

    data_dict = {}

    for i, token in enumerate(tokens): 
        token_prices = base_trend + reversion_scale* sample_OUP(num_periods)

        opens = token_prices + np.abs(np.min(token_prices)) + 1

        # Store in dictionary
        data_dict[token] = {
            "time": np.arange(num_periods),
            "open": opens,
        }

    return data_dict

def save_to_hdf5(data, filename="statarb_data.h5"):
    """
    Saves the OHLC trading data to an HDF5 file.
    
    Args:
        data (dict): Dictionary containing OHLC data for each token.
        filename (str): Name of the HDF5 file to save.
    """
    with h5py.File(filename, "w") as h5file:
        for token, ohlc_data in data.items():
            group = h5file.create_group(token)  # Create a group for each token
            for key, values in ohlc_data.items():
                group.create_dataset(key, data=values)  # Save data in the group


for i in tqdm(range(0,200),desc='Generating Arbritrage Data',unit=' episodes'):
    tokens = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    data = generate_trending_cointegrated_data(tokens, num_periods=2000,noise=0.2)
    save_to_hdf5(data, f"algo-data/episode_{i}.h5")

