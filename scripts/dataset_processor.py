import os
import h5py
import numpy as np
from tqdm import tqdm

def split_and_calculate_log_returns(input_filename, data_dir='data/', train_ratio=0.8):
    """ 
    Splits the dataset into train/test, then calculates log returns
    on each split file. 
    """
    # Ensure trailing slash
    if not data_dir.endswith('/'):
        data_dir += '/'
        
    full_path = os.path.join(data_dir, input_filename)
    
    # Derive output filenames
    base_name, ext = os.path.splitext(input_filename)
    train_file = os.path.join(data_dir, f"{base_name}_train{ext}")
    test_file  = os.path.join(data_dir, f"{base_name}_test{ext}")
    
    # 1) Split into train/test
    with h5py.File(full_path, "r") as hf_in, \
         h5py.File(train_file, "w") as hf_train, \
         h5py.File(test_file,  "w") as hf_test:
         
        tokens = list(hf_in.keys())  # e.g. ['BTCUSDT', 'ETHUSDT']
        
        for token in tqdm(tokens, desc="Splitting into train/test", unit="token"):
            group_in = hf_in[token]
            group_train = hf_train.create_group(token)
            group_test  = hf_test.create_group(token)
            
            for dset_key in group_in.keys():
                data = group_in[dset_key][:]
                
                # Split index
                n = data.shape[0]
                split_idx = int(n * train_ratio)

                # Write train split
                group_train.create_dataset(
                    dset_key, data=data[:split_idx],
                    maxshape=(None,), compression="gzip"
                )
                # Write test split
                group_test.create_dataset(
                    dset_key, data=data[split_idx:],
                    maxshape=(None,), compression="gzip"
                )

    # 2) Calculate log returns on the train file
    _calculate_log_returns(train_file)

    # 3) Calculate log returns on the test file
    _calculate_log_returns(test_file)

def _calculate_log_returns(h5_filepath):
    """
    Appends log returns to 'open','close','high','low' in place,
    removing the first element in each of those arrays.
    """
    with h5py.File(h5_filepath, "a") as f:
        tokens = list(f.keys())
        
        for token in tqdm(tokens, desc=f"Calculating log returns for {os.path.basename(h5_filepath)}", unit="token"):
            for timeseries_key in list(f[token].keys()):
                if timeseries_key in ['open','close','high','low']:
                    timeseries = f[token][timeseries_key][:]
                    
                    # Avoid length issues if there's fewer than 2 data points
                    if len(timeseries) < 2:
                        continue
                    
                    # Compute log returns
                    log_returns = np.log(timeseries[1:] / timeseries[:-1])
                    
                    # Replace original dataset (dropping 1st data point)
                    del f[token][timeseries_key]
                    f[token].create_dataset(timeseries_key, data=timeseries[1:], compression="gzip")
                    
                    # If a log-return dataset already exists, overwrite
                    log_key = "log_return_" + str(timeseries_key)
                    if log_key in f[token]:
                        del f[token][log_key]
                    f[token].create_dataset(log_key, data=log_returns, compression="gzip")
                
                else:
                    # For other datasets, just remove the first element
                    # (to keep them in sync with open/high/low/close)
                    arr = f[token][timeseries_key][:]
                    if len(arr) > 1:
                        del f[token][timeseries_key]
                        f[token].create_dataset(timeseries_key, data=arr[1:], compression="gzip")

if __name__ == "__main__":


    # Modify or loop over your entire dataset list as needed.
    split_and_calculate_log_returns("dataset_60000_1h.h5", data_dir="data/", train_ratio=0.8)
