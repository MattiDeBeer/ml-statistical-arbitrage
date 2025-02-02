# scripts/train.py
import sys
from scripts.plot import evaluate_and_plot
sys.path.append("../")
from envs.stable_baseline_enviroment import DQNTradingEnv
from trading_models.DQN_Model import train_dqn

def main():

    # 2) Create environment
    env = DQNTradingEnv(10)

    # 3) Train agent
    # If using stable-baselines3 DQN:
    model = train_dqn(env, total_timesteps=100000, verbose=1)

    # 4) Evaluate
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ , info = env.step(action)
        print(f"Action: {action}")
        print(f"Obvs: {obs}")
        print(f"Reward {reward}")
        input('>')
        total_reward += reward
    
    print("Test episode total reward:", total_reward)
    
    #5) Plot
    #evaluate_and_plot(env, model)

if __name__ == "__main__":
    main()

