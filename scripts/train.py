# scripts/train.py

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../")
from envs.stable_baseline_enviroment import DQNTradingEnv
from trading_models.DQN_Model import train_dqn



def main():

    # 2) Create environment
    env = DQNTradingEnv(10)

    # 3) Train agent
    # If using stable-baselines3 DQN:
    model = train_dqn(env, total_timesteps=500000, verbose=1)

    # 4) Evaluate
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    action_arr = []
    price_arr = []
    step_arr = []
    step = 0
    
    while not done:
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ , info = env.step(action)
        total_reward += reward
        action_arr.append(info['action_type'])
        price_arr.append(obs[-2])
        step+= 1
        step_arr.append(step)
        
    
    print("Test episode total reward:", total_reward)
    
    #5) Plot
    # Plot price vs step
    action_arr = np.array(action_arr)
    step_arr = np.array(step_arr)
    price_arr = np.array(price_arr)
    plt.plot(step_arr, price_arr, label='Price')
    
    # Mark buy actions with triangles (1 in action_arr)
    buy_steps = step_arr[action_arr == 1]
    buy_prices = price_arr[action_arr == 1]
    plt.scatter(buy_steps, buy_prices, marker='^', color='green', label='Buy', zorder=5)
    
    # Mark sell actions with circles (-1 in action_arr)
    sell_steps = step_arr[action_arr == -1]
    sell_prices = price_arr[action_arr == -1]
    plt.scatter(sell_steps, sell_prices, marker='o', color='red', label='Sell', zorder=5)
    
    # Add labels and legend
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend()
    
    # Show plot
    plt.show()
    
    
if __name__ == "__main__":
    main()

