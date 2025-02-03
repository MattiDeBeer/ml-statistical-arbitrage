#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:12:18 2025

@author: matti
"""

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def train_dqn(env, total_timesteps=10000, verbose=1, **kwargs):
    """
    Trains a DQN on the given environment using stable-baselines3.

    Parameters
    ----------
    env : gym.Env
        The trading environment.
    total_timesteps : int
        Total number of timesteps to train for.
    verbose : int
        Verbosity level: 0 no output, 1 info, 2 debug.

    Returns
    -------
    model : stable_baselines3.DQN
        The trained DQN model.
    """
    # Convert the environment to a vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    # Create DQN model
    # For more options, see https://stable-baselines3.readthedocs.io/
    model = DQN(
        policy="MlpPolicy", 
        env=vec_env, 
        verbose=verbose,
        device = "auto",
        # Some hyperparameters you might want to tune:
        learning_rate=1e-5,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=0.99,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        **kwargs
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    return model
