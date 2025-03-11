# scripts/train.py

import sys

sys.path.append("../")
from rl_models.stable_baseline_DQN_Model import DqnModelCont, DqnModelDict
from envs.rl_enviroments import RlTradingEnvContinious, RlTradingEnvDict, TestEnv, RlTradingEnvBTC



Model = DqnModelDict(RlTradingEnvBTC)

Model.train(10000)

#%%
Model.plot_episode()
