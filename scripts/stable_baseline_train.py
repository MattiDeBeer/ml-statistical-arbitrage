# scripts/train.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from rl_models.stable_baseline_DQN_Model import DqnModelCont, DqnModelDict
from envs.rl_enviroments import RlTradingEnvContinious, RlTradingEnvDict, TestEnv



Model = DqnModelDict(RlTradingEnvDict)

Model.train(10000)

Model.plot_episode()
