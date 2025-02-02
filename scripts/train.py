# scripts/train.py

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
from envs.stable_baseline_enviroment import DQNTradingEnv
from trading_models.DQN_Model import train_dqn



def main():

    # 2) Create environment
    env = DQNTradingEnv(20)

    # 3) Train agent
    # If using stable-baselines3 DQN:
    model = train_dqn(env, total_timesteps=100000, verbose=1)

    # 4) Evaluate
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    action_arr = []
    price_arr = []
    step_arr = []
    
    while not done:
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ , info = env.step(action)
        total_reward += reward
        action_arr.append(info['action'])
        price_arr.append(info['price'])
        step_arr.append(info['step'])
        
    
    print("Test episode total reward:", total_reward)
    
    #5) Plot
    price_arr = np.array(price_arr)
    action_arr = np.array(action_arr)
    buy_array_mask = action_arr == 1
    sell_array_mask = action_arr == -1
    plt.plot(step_arr, price_arr)   
    plt.scatter(step_arr[buy_array_mask],price_arr[buy_array_mask], 'O')
    plt.scatter(step_arr[sell_array_mask],price_arr[sell_array_mask], 'X')
    plt.show()
    
    
if __name__ == "__main__":
    main()

