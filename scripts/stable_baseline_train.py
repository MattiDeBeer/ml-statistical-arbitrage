# scripts/train.py

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from rl_models.stable_baseline_DQN_Model import DqnModel
from envs.rl_enviroments import RlTradingEnv, TestEnv


Model = DqnModel(RlTradingEnv)

Model.train(100000)

Model.plot_episode()