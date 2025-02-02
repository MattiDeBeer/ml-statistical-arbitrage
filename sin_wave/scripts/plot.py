import matplotlib.pyplot as plt

def evaluate_and_plot(env, model):
    obs = env.reset()
    done = False

    # Lists to record the data
    prices = []
    steps = []
    actions = []  # 0: Sell, 1: Hold, 2: Buy

    # Evaluate the episode
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        price = env.df.iloc[env.current_step]['Close']
        prices.append(price)
        steps.append(env.current_step)
        actions.append(action)

    # Plot the sine wave (Close prices)
    plt.figure(figsize=(12, 6))
    plt.plot(env.df['Close'].values, label='Close Price', color='black', linewidth=1.5)

    # Determine markers for each action
    buy_steps = [s for s, a in zip(steps, actions) if a == 2]
    sell_steps = [s for s, a in zip(steps, actions) if a == 0]
    hold_steps = [s for s, a in zip(steps, actions) if a == 1]

    # Get prices at these steps for plotting markers
    buy_prices = env.df.iloc[buy_steps]['Close'].values if buy_steps else []
    sell_prices = env.df.iloc[sell_steps]['Close'].values if sell_steps else []
    # (Optionally, you might not mark holds if they are too frequent)
    hold_prices = env.df.iloc[hold_steps]['Close'].values if hold_steps else []

    # Plot markers
    plt.scatter(buy_steps, buy_prices, marker='^', color='green', s=100, label='Buy')
    plt.scatter(sell_steps, sell_prices, marker='v', color='red', s=100, label='Sell')
    plt.scatter(hold_steps, hold_prices, marker='x', color='blue', s=100, label='Hold')
    # Uncomment the following line if you want to mark holds as well:
    # plt.scatter(hold_steps, hold_prices, marker='o', color='blue', s=50, label='Hold')

    plt.title("Trading Actions on Sine Wave")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Example usage (assuming you have a trained model and your environment is set up):
# from agents.dqn_agent import train_dqn
# model = train_dqn(env, total_timesteps=20000, verbose=1)
# evaluate_and_plot(env, model)

