import gym
import numpy as np
from gym import spaces
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, window_size=5):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.window_size = window_size

        # The observation will be a flattened vector of the window plus shares held.
        # 3 features (Close, RSI, MACD) per period and 1 extra feature for shares held.
        obs_size = window_size * 3 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Sell, 1: Hold, 2: Buy

    def _get_window_data(self):
        # Get data from the past 'window_size' steps.
        # If there isn’t enough history, pad with the first row.
        start = max(0, self.current_step - self.window_size + 1)
        window = self.df.iloc[start:self.current_step + 1][['Close', 'rsi', 'macd']].values
        if len(window) < self.window_size:
            # Repeat the first available row if needed for padding
            pad = np.tile(window[0], (self.window_size - len(window), 1))
            window = np.vstack([pad, window])
        return window

    def _get_observation(self):
        window_data = self._get_window_data()
        obs = window_data.flatten()  # flatten the (window_size, 3) array
        obs = np.concatenate([obs, [float(self.shares_held)]])  # add shares held as last element
        return obs.astype(np.float32)

    def step(self, action):
        # ... (the rest of your step logic remains unchanged)
        # Make sure to call self._get_observation() at the end to return the updated observation.
        # (Your current step, reward, info, etc., remain the same.)
        old_price = self.df.iloc[self.current_step]['Close']
        old_portfolio_value = self.balance + self.shares_held * old_price

        if action == 0:  # Sell all
            self.balance += self.shares_held * old_price
            self.shares_held = 0
        elif action == 2:  # Buy as many as possible
            max_shares = self.balance // old_price
            self.shares_held += max_shares
            self.balance -= max_shares * old_price

        self.current_step += 1
        done = self.current_step >= self.n_steps
        if done:
            self.current_step = self.n_steps - 1

        new_price = self.df.iloc[self.current_step]['Close']
        new_portfolio_value = self.balance + self.shares_held * new_price
        if old_portfolio_value > 0 and new_portfolio_value > 0:
            reward = np.log(new_portfolio_value) - np.log(old_portfolio_value)
        else:
            reward = -1.0

        obs = self._get_observation()
        info = {"old_portfolio_value": old_portfolio_value, "new_portfolio_value": new_portfolio_value}

        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        return self._get_observation()

    def render(self, mode='human'):
        row = self.df.iloc[self.current_step]
        price = row['Close']
        value = self.balance + self.shares_held * price
        print(f"Step: {self.current_step}, Price: {price:.2f}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, Value: {value:.2f}")
