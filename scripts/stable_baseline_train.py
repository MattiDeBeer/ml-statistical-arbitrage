# scripts/train.py

from models.dqn_models import DqnModel, PairsDqnModel
from models.feature_extractors import SingleTokenFeatureExtractor, PairsFeatureExtractor
from envs.rl_enviroments import RlTradingEnvToken, RlTradingEnvSin, RlTradingEnvPairs
import numpy as np


single_config = {
    
    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvSin, #or RlTradingEnvToken
    "episode_length": 100,
    "token" : 'BTCUSDT',
    "timeseries_obs" : {'open' : (10,-np.inf,np.inf)}, 
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 2},
    "verbose" : False,
    "transaction_percentage" : 0.001,
    "token_pair" : ("BTCUSDT","ETHUSDT"),
    
    ### Feature Extractor Config ###
    "feature_extractor_class" : SingleTokenFeatureExtractor,
    "combiner_layers" : [10,8,5],
    "disc_layers" : [3,2],
    "lstm_hidden_size" : 5,
    "compile_flag" : False,
    "feature_dim" : 5,
    
    ### DQN Config ###
    "learning_rate" : 1e-3,
    "buffer_size" : 1000,
    "learning_starts": 500,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 250,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.7,
    "q_net_layers" : [10],
    "verbose_level" : 2
}

pairs_config = {
    
    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvPairs,
    "episode_length": 1000,
    "timeseries_obs" : {'z_score' : (10,-np.inf,np.inf),'volume' : (10,-np.inf,np.inf)},
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 2},
    "indicator_obs" : {'adfuller' : (0,1), 'coint_p_value' : (0,1)},
    "verbose" : False,
    "transaction_percentage" : 0,
    "token_pair" : ("BTCUSDT","ETHUSDT"),
    "z_score_context_length" : 150,
    "coint_context_length" : 150,
    "log" : True,
    "dataset_file": "data/dataset_60000_1h_train.h5",
    "test_dataset": "data/dataset_60000_1h_test.h5",

    
    
    ### Feature Extractor Config ###
    "feature_extractor_class" : PairsFeatureExtractor,
    "combiner_layers" : [10],
    "disc_layers" : [5,4],
    "indicator_layers" : [3,2],
    "lstm_hidden_size" : 10,
    "compile_flag" : False,
    "feature_dim" : 8,
    "GPU_available" : False,
    
    ### DQN Config ###
    "learning_rate" : 1e-3,
    "buffer_size" : 1000,
    "learning_starts": 50,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 500,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.5,
    "q_net_layers" : [32,32],
    "verbose_level" : 0
}

### THIS TRAINS AND PLOTS AN EPISODE FOR SINGLE TOKEN TRADING ###

#Model = DqnModel(single_config)
#Model.train(100000)
#Model.plot_episode()

#################################################################

### THIS TRAINS AND PLOTS AN EPISODE FOR PAIRS TRADING ###

Model = PairsDqnModel(pairs_config)
Model.train(100)
Model.save("test")
Model.plot_episode()

### #############################################################
