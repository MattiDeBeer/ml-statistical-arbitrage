# scripts/train.py

from models.stable_baseline_DQN_Model import DqnModel
from envs.rl_enviroments import RlTradingEnvToken, RlTradingEnvSin
import numpy as np

config = {
    
    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvSin,
    "episode_length": 500,
    "token" : 'BTCUSDT',
    "timeseries_obs" : {'open' : (10,-np.inf,np.inf), 'close' : (5,-np.inf, np.inf)},
    #"indicator_obs" : {'adfuller1' : (0,1)},
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 2},
    "verbose" : True,
    "transaction_percentage" : 0,
    
    ### Feature Extractor Config ###
    "combiner_layers" : [10,20,5],
    "disc_layers" : [3,2],
    #"indicator_layers" : [10,10],
    "lstm_hidden_size" : 10,
    "compile_flag" : False,
    
    ### DQN Config ###
    "learning_rate" : 1e-3,
    "buffer_size" : 10000,
    "learning_starts": 500,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 500,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.5,
}

Model = DqnModel(config)

Model.train(10)

#%%

Model.plot_episode()
