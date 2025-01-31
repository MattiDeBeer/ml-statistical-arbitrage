# scripts/train.py

from data_generation.synthetic_data import generate_sine_wave_data
from envs.trading_env import TradingEnv
from agents.dqn_agent import train_dqn  # or DQNAgent if custom

def main():
    # 1) Generate data
    df = generate_sine_wave_data(
        n_cycles=10, 
        points_per_cycle=360, 
        noise_factor=0.05, 
        seed=42
    )

    # 2) Create environment
    env = TradingEnv(df, initial_balance=10000)

    # 3) Train agent
    # If using stable-baselines3 DQN:
    model = train_dqn(env, total_timesteps=20000, verbose=1)

    # 4) Evaluate
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    
    print("Test episode total reward:", total_reward)

if __name__ == "__main__":
    main()

