import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    """
    A trading environment with a discrete action space:
        0 = Sell (liquidate shares),
        1 = Hold,
        2 = Buy (use all available balance).
    Reward: log return of the portfolio value.
    """
    
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        # Store the price data and any indicators
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        
        # Environment state
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        
        # Define action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # Example: [Close, RSI, MACD, shares_held]
        # You can add more features or a time window as needed
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        
    def _get_observation(self):
        """
        Return the current state as a numpy array.
        """
        row = self.df.iloc[self.current_step]
        
        # E.g., [Close, RSI, MACD, position_size]
        obs = np.array([
            row['Close'],
            row['rsi'],
            row['macd'],
            float(self.shares_held)
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        # 1) Calculate the old portfolio value
        old_price = self.df.iloc[self.current_step]['Close']
        old_portfolio_value = self.balance + self.shares_held * old_price
        
        # 2) Apply the action
        # Action: 0 = sell, 1 = hold, 2 = buy
        if action == 0:
            # Sell all
            self.balance += self.shares_held * old_price
            self.shares_held = 0
        elif action == 2:
            # Buy as many shares as we can afford
            # (This is simplistic and does not consider fractional shares, fees, etc.)
            max_shares = self.balance // old_price
            self.shares_held += max_shares
            self.balance -= max_shares * old_price
        # if action == 1, do nothing (hold)
        
        # 3) Advance to the next step
        self.current_step += 1
        
        # If we run off the end of the dataset, episode ends
        if self.current_step >= self.n_steps:
            self.current_step = self.n_steps - 1
            done = True
        else:
            done = False

        # 4) Compute new portfolio value at the new price
        new_price = self.df.iloc[self.current_step]['Close']
        new_portfolio_value = self.balance + self.shares_held * new_price
        
        # 5) Compute reward as log return
        if old_portfolio_value > 0 and new_portfolio_value > 0:
            reward = np.log(new_portfolio_value) - np.log(old_portfolio_value)
        else:
            # If either value is 0 or negative, handle carefully
            reward = -1.0  # Or some penalty; handle as needed
        
        # 6) Get next observation
        obs = self._get_observation()
        
        # Info dictionary can carry additional data for debugging
        info = {
            "old_portfolio_value": old_portfolio_value,
            "new_portfolio_value": new_portfolio_value
        }
        
        return obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        
        return self._get_observation()

    def render(self, mode='human'):
        """
        Render the environment or provide debug info.
        """
        row = self.df.iloc[self.current_step]
        price = row['Close']
        value = self.balance + self.shares_held * price
        print(f"Step: {self.current_step}, Price: {price:.2f}, "
              f"Balance: {self.balance:.2f}, Shares: {self.shares_held}, "
              f"Value: {value:.2f}")

