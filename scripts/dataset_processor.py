import os
import h5py
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax.scipy.stats import norm
import cupy as cp
from cupy.linalg import lstsq
import shutil
import statsmodels.api as sm



def split_datasets(filename, train_ratio=0.7, val_ratio=0.15):
    """ 
    Splits the dataset into train, validation, and test sets,
    then (optionally) processes each split (e.g., calculates log returns).
    
    Parameters
    ----------
    filename : str
        The path to the original dataset file.
    train_ratio : float, default=0.6
        Proportion of data to use for training.
    val_ratio : float, default=0.2
        Proportion of data to use for validation.
        (Test set will use 1 - train_ratio - val_ratio of the data.)
    """
    # Determine directory and file names
    input_filename = filename.split('/')[-1]
    data_dir = '/'.join(filename.split('/')[:-1]) + '/'
    if not data_dir.endswith('/'):
        data_dir += '/'
    full_path = os.path.join(data_dir, input_filename)
    
    # Derive output filenames
    base_name, ext = os.path.splitext(input_filename)
    train_file = os.path.join(data_dir, f"{base_name}_train{ext}")
    val_file   = os.path.join(data_dir, f"{base_name}_val{ext}")
    test_file  = os.path.join(data_dir, f"{base_name}_test{ext}")
    
    # Open the original file and create new files for each split
    with h5py.File(full_path, "r") as hf_in, \
         h5py.File(train_file, "w") as hf_train, \
         h5py.File(val_file, "w") as hf_val, \
         h5py.File(test_file, "w") as hf_test:
         
        tokens = list(hf_in.keys())  # e.g., ['BTCUSDT', 'ETHUSDT']
        
        for token in tqdm(tokens, desc="Splitting dataset", unit="token", leave=False):
            group_in = hf_in[token]
            group_train = hf_train.create_group(token)
            group_val   = hf_val.create_group(token)
            group_test  = hf_test.create_group(token)
            
            for dset_key in group_in.keys():
                data = group_in[dset_key][:]
                n = data.shape[0]
                # Compute split indices
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                
                # Write train split
                group_train.create_dataset(
                    dset_key, data=data[:train_end],
                    maxshape=(None,), compression="gzip"
                )
                # Write validation split
                group_val.create_dataset(
                    dset_key, data=data[train_end:val_end],
                    maxshape=(None,), compression="gzip"
                )
                # Write test split
                group_test.create_dataset(
                    dset_key, data=data[val_end:],
                    maxshape=(None,), compression="gzip"
                )
    
                
def calculate_log_returns(timeseries1):
    """
    calculates the log returns for a timeseries
    """
    log_returns = np.diff(np.log(timeseries1))
    return log_returns
   
                        
def jax_adf_test(series):
    """
    Performs a GPU optimized adf test.
    paramaters:
    series (np array): The series you which to calculate the test for
    returns:
        test_stat: The test statitsics
    p_value (float): The p value of the test
     
   """
    series = jnp.asarray(series)
    delta_y = jnp.diff(series)
    lagged_y = series[:-1]
     
    # Compute OLS regression (least squares)
    beta = jnp.sum(lagged_y * delta_y) / jnp.sum(lagged_y ** 2)
    residuals = delta_y - beta * lagged_y
    se = jnp.sqrt(jnp.sum(residuals ** 2) / (len(series) - 2))
     
    # Compute test statistic (similar to adfuller)
    test_stat = beta / (se / jnp.sqrt(jnp.sum(lagged_y ** 2)))
     
    # Get critical value (normal approximation for speed)
    p_value = 2 * (1 - norm.cdf(jnp.abs(test_stat)))
     
    return p_value

def cupy_cointegration_test(y1, y2):
    """
    Performs a GPU optimized cointegraiton test using CuPy
    parameters:
    y1 (np array): The first series you whish to test
    y2 (np.array): The second series you whish to test
    return:
    test_stat: The test statistics
    p_value (float): The p value of the test
   """
    y1, y2 = cp.asarray(y1), cp.asarray(y2)
    X = cp.vstack([y2, cp.ones(len(y2))]).T  # Add intercept
    beta, _, _, _ = lstsq(X, y1)  # Solve OLS y1 ~ y2
    
    residuals = y1 - (X @ beta)  # Compute residuals
    test_stat, p_value = jax_adf_test(cp.asnumpy(residuals))  # Apply fast ADF test
    
    return p_value          

def numpy_cointegration_test(y1, y2):
    """
    parameters:
    y1 (np array): The first series you whish to test
    y2 (np.array): The second series you whish to test
    return:
    test_stat: The test statistics
    p_value (float): The p value of the test
   """
    y1, y2 = np.asarray(y1), np.asarray(y2)
    X = np.vstack([y2, np.ones(len(y2))]).T  # Add intercept
    beta, _, _, _ = np.linalg.lstsq(X, y1)  # Solve OLS y1 ~ y2
    
    residuals = y1 - (X @ beta)  # Compute residuals
    p_value = jax_adf_test(residuals)  # Apply fast ADF test
    
    return p_value            

def calc_hedge_ratio(ts1, ts2):
    """
    Calculate the hedge ratio using OLS regression.
    
    :param ts1: numpy array or list, time series of asset 1 (dependent variable)
    :param ts2: numpy array or list, time series of asset 2 (independent variable)
    :return: hedge ratio (float)
    """

    # Add a constant to the independent variable (ts2) for OLS regression
    ts2 = sm.add_constant(ts2)

    # Fit OLS regression: ts1 = beta * ts2 + error
    model = sm.OLS(ts1, ts2).fit()

    # Hedge ratio is the slope (beta coefficient)
    hedge_ratio = model.params[1]

    return hedge_ratio                     
                        
def calc_adfuller(timeseries, context_length,key):
    adf = []
    for i in tqdm( range(context_length,len(timeseries)), desc=f"Calculating adf for {key}", unit = " datapoint", leave=False):
        adf.append(jax_adf_test(timeseries[i-context_length:i]))

    return(np.array(adf))

def calculate_z_scores(timeseries1, timeseries2,key,context_length=10):
    z_scores = []

    assert len(timeseries2) == len(timeseries2), "Timeseries must have the same length"
    for i in tqdm( range (context_length,len(timeseries2)), desc=f"calculating z scores for {key}", unit=" datapoint",leave=False):
        ts1 = timeseries1[i-context_length:i]
        ts2 = timeseries2[i-context_length:i]
        hedge_ratio = calc_hedge_ratio(ts1,ts2)
        spread = ts1- hedge_ratio * ts2
        z_score = (spread[-1] - np.mean(spread)) / (np.std(spread))
        z_scores.append(z_score)

    return z_scores

def calculate_and_append_log_returns(dataset_file, processed_dataset_file,log_return_keys):
    for key in dataset_file.keys():
        if key in log_return_keys:
            timeseries = dataset_file[key][:]
            log_return = calculate_log_returns(timeseries)
            append_dataset(f"log_return_{key}",processed_dataset_file,log_return)

def calculate_and_append_adfuler(dataset_file, processed_dataset_file,adfuller_keys,context_length=10):
    for key in dataset_file.keys():
        if key in adfuller_keys:
            timeseries = dataset_file[key][:]
            adfuller = calc_adfuller(timeseries,context_length,f"adfuller_{key}_{context_length}")

            append_dataset(f"adfuller_{key}_{context_length}",processed_dataset_file,adfuller)

def calculate_and_append_z_score(dataset_file, processed_dataset_file,pair,z_score_keys,context_length=10):
    token1,token2 = pair.split('-')
    for key in dataset_file[token1]:
        if key in dataset_file[token2].keys() and key in z_score_keys:
            timeseries1 = dataset_file[token1][key][:]
            timeseries2 = dataset_file[token2][key][:]

            z_scores = calculate_z_scores(timeseries1,timeseries2,f"z_score_{pair}_{key}_{context_length}",context_length=context_length)

            create_group_if_not_exists(pair,processed_dataset_file)
            append_dataset(f"z_score_{key}_{context_length}", processed_dataset_file[pair], z_scores)

def calculate_and_append_coint_p_values(dataset_file, processed_dataset_file, pair, coint_p_value_keys, context_length=10,GPU = False):
    token1, token2 = pair.split('-')
    for key in dataset_file[token1]:
        if key in dataset_file[token2].keys() and key in coint_p_value_keys:
            timeseries1 = dataset_file[token1][key][:]
            timeseries2 = dataset_file[token2][key][:]

            coint_p_values = []
            for i in tqdm(range(context_length, len(timeseries1)), desc=f"Calculating coint p-values for {pair}_{key}_{context_length}", unit=" datapoint", leave=False):
                if GPU:
                    coint_p_value = cupy_cointegration_test(timeseries1[i-context_length:i], timeseries2[i-context_length:i])
                else:
                    coint_p_value = numpy_cointegration_test(timeseries1[i-context_length:i], timeseries2[i-context_length:i])

                coint_p_values.append(coint_p_value)

            coint_p_values = np.array(coint_p_values)

            create_group_if_not_exists(pair, processed_dataset_file)
            append_dataset(f"coint_p_value_{key}_{context_length}", processed_dataset_file[pair], coint_p_values)

def get_token_pairs(tokens):
    token_pairs = []
    for i in range(0,len(tokens)-1):
        for j in range(i+1,len(tokens)):
            token_pairs.append(f"{tokens[i]}-{tokens[j]}")
    
    return token_pairs

def create_group_if_not_exists(key,dataset_file):
    if key not in dataset_file:
        dataset_file.create_group(key)

def append_dataset(dataset_name,file,data):
    if dataset_name in file:
        del file[dataset_name]

    file.create_dataset(dataset_name, data=data, compression="gzip")

def normalise_timeseries_lengths(file):

    min_length = float('inf')
    for token in file.keys():
        for key in file[token].keys():
            length = len(file[token][key])
            if length < min_length:
                min_length = length

    for token in file.keys():
        for key in file[token].keys():
            data = file[token][key][:]
            if len(data) > min_length:
                del file[token][key]
                file[token].create_dataset(key, data=data[-min_length:], compression="gzip")
                

if __name__ == "__main__":

    dataset_filename = "data/dataset_60000_1h.h5"
    processed_dataset_filename = "data/processed_dataset_60000_1h.h5"
    
    # Delete processed file if it exists
    if os.path.exists(processed_dataset_filename):
        os.remove(processed_dataset_filename)

    #create a copy of the dataset file
    shutil.copy2(dataset_filename, processed_dataset_filename)

    log_return_keys = ['open','close','high','low']
    adfuller_keys = ['open']
    coint_p_value_keys = ['open']
    z_score_keys = ['open']
    coint_p_value_keys = ['open']

    context_lengths = [50,100,200,400]

    for context_length in context_lengths:

        with h5py.File(dataset_filename, "r") as dataset_file, h5py.File(processed_dataset_filename, "a") as processed_dataset_file :

            dataset_keys = list(dataset_file.keys())
            tokens = [item for item in dataset_keys if '-' not in item]
            
            for token in tokens:

                calculate_and_append_log_returns(dataset_file[token],processed_dataset_file[token],log_return_keys)

                calculate_and_append_adfuler(dataset_file[token],processed_dataset_file[token],adfuller_keys,context_length=context_length)

            token_pairs = get_token_pairs(tokens)
            
            for token_pair in token_pairs:

                calculate_and_append_z_score(dataset_file,processed_dataset_file,token_pair,z_score_keys,context_length = context_length)

                calculate_and_append_coint_p_values(dataset_file,processed_dataset_file,token_pair,coint_p_value_keys,context_length=context_length,GPU=False)

        
    with h5py.File(processed_dataset_filename, "a") as processed_dataset_file :
        normalise_timeseries_lengths(processed_dataset_file)

    split_datasets(processed_dataset_filename)


            
