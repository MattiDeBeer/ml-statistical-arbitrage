# scripts/train.py

from models.stable_baseline_DQN_Model import DqnModel
from envs.rl_enviroments import RlTradingEnvToken, RlTradingEnvSin
import numpy as np

config = {
    
    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvSin,
    "episode_length": 500,
    "token" : 'BTCUSDT',
    "continious_dim" : 20,
    "continious_obs" : {'open' : (-np.inf,np.inf), 'close' : (-np.inf, np.inf)},
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 2},
    "verbose" : True,
    
    ### Feature Extractor Config ###
    "combiner_hidden" : [10,20],
    "disc_hidden" : [3],
    "lstm_hidden_size" : 30,
    "disc_out_dim" : 5,
    "compile_flag" : False

}

Model = DqnModel(config)

Model.train(10000)

Model.plot_episode()
