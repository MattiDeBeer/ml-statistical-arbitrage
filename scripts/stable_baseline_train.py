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
    "combiner_hidden" : [10,20],
    "disc_hidden" : [3],
    "lstm_hidden_size" : 30,
    "disc_out_dim" : 5,
    "compile_flag" : False

}

Model = DqnModel(config)

Model.train(100000)

#%%

Model.plot_episode()
