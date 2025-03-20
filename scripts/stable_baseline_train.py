# scripts/train.py

from models.dqn_models import DqnModel, PairsDqnModel
from models.feature_extractors import SingleTokenFeatureExtractor, PairsFeatureExtractor
from envs.rl_enviroments import RlTradingEnvToken, RlTradingEnvSin, RlTradingEnvPairs, RlTradingEnvPairsExtendedActions, RlPretrainEnvSingleAction
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
    
    "run_id" : "0005",

    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvPairs,
    "episode_length": 800,
    "timeseries_obs" : {'z_score' : (10,-np.inf,np.inf)},
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 2},
    "indicator_obs" : {'adfuller' : (0,1), 'coint_p_value' : (0,1)},
    "verbose" : False,
    "transaction_percentage" : 0,
    "token_pair" : ("BTCUSDT","ETHUSDT"),
    "z_score_context_length" : 100,
    "coint_context_length" : 100,
    "log" : True,
    "dataset_file": "data/processed_dataset_5000_1h_train.h5",
    "test_dataset": "data/processed_dataset_5000_1h_test.h5",
    "algo_env" : RlTradingEnvPairs,
    "pretrain_env" : RlPretrainEnvSingleAction,
    
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
    "exploration_fraction": 0.7,
    "q_net_layers" : [64,64,64],
    "verbose_level" : 0,
    "tensorboard_log_file" : "tensorboard_log",
    "model_save_folder" : "saved_models"
}

pairs_config_extended_actions = {
    
    "run_id" : "0001",

    ### Enviroment Config ###
    "enviromentClass": RlTradingEnvPairsExtendedActions,
    "episode_length": 500,
    "timeseries_obs" : {},
    "discrete_obs" : {'is_bought' : 2, 'previous_action' : 3},
    "indicator_obs" : {'adfuller' : (0,1), 'coint_p_value' : (0,1), 'amount_bought': (0,np.inf), 'z_score': (-np.inf,np.inf)},
    "verbose" : False,
    "transaction_percentage" : 0,
    "token_pair" : ("BTCUSDT","ETHUSDT"),
    "z_score_context_length" : 100,
    "coint_context_length" : 100,
    "log" : True,
    "dataset_file": "data/processed_dataset_5000_1h_train.h5",
    "test_dataset": "data/processed_dataset_5000_1h_test.h5",
    "algo_env" : RlTradingEnvPairsExtendedActions,
    "pretrain_env" : RlTradingEnvPairsExtendedActions,
    
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
    "learning_starts": 200,
    "batch_size": 100,
    "gamma": 0.99,
    "target_update_interval": 500,
    "exploration_initial_eps": 0.25,
    "exploration_final_eps": 0.05,
    "exploration_fraction": 0.8,
    "q_net_layers" : [64,64,64],
    "verbose_level" : 0,
    "tensorboard_log_file" : "./tensorboard_log",
    "model_save_folder" : "saved_models"
}

### THIS TRAINS AND PLOTS AN EPISODE FOR SINGLE TOKEN TRADING ###

#Model = DqnModel(single_config)
#Model.train(100000)
#Model.plot_episode()

#################################################################

### THIS TRAINS AND PLOTS AN EPISODE FOR PAIRS TRADING ###

#Model = PairsDqnModel(pairs_config)
#Model.train(1,eval_frequency=5,eval_steps=5)
#Model.plot_episode(action_num=1)
#Model.save("test")
#

### #############################################################

### THIS TRAINS AND PLOTS AN EPISODE FOR PAIRS TRADING WITH EXTENDED ACTIONS ###

Model = PairsDqnModel(pairs_config)
Model.force_pretrain_qnet(dataset_size=10000,steps=1000,verbose_freq = 100)
while Model.pretrain_eval_episode(30) <= 0:
    Model.pre_train(200,eval_frequency=10,eval_steps=5)

average = Model.pretrain_eval_episode(30)
print(f"Model Pretraining complete with an average reward of {average}")

while Model.eval_episode(30,algo=True) <= 0:
    Model.train_algo(200,eval_frequency=100,eval_steps=5)

average = Model.eval_episode(30,algo=True)
print(f"Model Pretraining complete with an average reward of {average}")

while Model.eval_episode(30,algo=True) <= 0:
    Model.train(200,eval_frequency=100,eval_steps=5)

average = Model.eval_episode(30,algo=True)
print(f"Model Pretraining complete with an average reward of {average}")

Model.save(pairs_config['run_id_final'])
#Model.save("test")
#

### #############################################################
