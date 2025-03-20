from scripts.dataset_processor import *

if __name__ == "__main__":
        
    for i in tqdm( range(0,100), desc = 'Processing files', unit=' file'):

        dataset_filename = f"algo-data/episode_{i}.h5"
        processed_dataset_filename = f"algo-data/processed_episode_{i}.h5"
        
        # Delete processed file if it exists
        if os.path.exists(processed_dataset_filename):
            os.remove(processed_dataset_filename)

        #create a copy of the dataset file
        shutil.copy2(dataset_filename, processed_dataset_filename)

        log_return_keys = ['open']
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

