import h5py
import numpy as np
from tqdm import tqdm

def calculate_log_returns(input_filename, dir = 'data/'):
    # Open the input HDF5 file in read mode
    with h5py.File(dir+input_filename, 'a') as f:
        for token in tqdm(list(f.keys()), desc="Calculating dataset Log returns", unit="token"):
            for timeseries_key in list(f[token].keys()):
                if timeseries_key in ['open','close','high','low']:
                    timeseries = f[token][timeseries_key][:]
                    log_returns = np.log(timeseries[1:]/timeseries[:-1])
                    del f[token][timeseries_key]  
                    f[token][timeseries_key] = timeseries[1:]

                    if "log_return_" + str(timeseries_key) in f[token]:
                        del f[token]["log_return_" + str(timeseries_key)]
                        f[token]["log_return_" + str(timeseries_key)] = log_returns
                    else:
                        f[token].create_dataset("log_return_" + str(timeseries_key), data=log_returns)
                else:
                    timeseries = f[token][timeseries_key][:]
                    del f[token][timeseries_key]  
                    f[token][timeseries_key] = timeseries[1:]
       






if __name__ == "__main__":
    """
    filenames = ['dataset_100000_15m.h5',  'dataset_100000_1m.h5',  'dataset_100000_5m.h5',  'dataset_10000_4h.h5',  'dataset_60000_1h.h5']
    for filename in filenames:
        print(f"Processing: {filename}")
        calculate_log_returns('dataset_10000_4h.h5')
    """
    calculate_log_returns('dataset_100000_1m.h5')
