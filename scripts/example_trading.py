from envs.binance_trading_enviroment import BinanceTradingEnv
from collections import defaultdict
import matplotlib.pyplot as plt

env = BinanceTradingEnv()
env.load_token_dataset()
env.load_token_dataset(filename='dataset_100000_1m.h5',directory = 'data/')


env.get_token_episode(['BTCUSDT','ETHUSDT'],1000)

#set parameters
done = False
context_length = 100

#Make sure time is stepped forward a sufficient length, to allow for enough previous datapoints
env.step_time(context_length+1)

#set the env start money
start_money = 100
env.money = 100

env.transacrion_fee_percentage = 0.001

is_bought = False

logging_dict = defaultdict(list)

while not done:
    
    #fetch the current prices
    current_btc_price = env.get_current_price('BTCUSDT')
    current_eth_price = env.get_current_price('ETHUSDT')
    
    #add to logging dict
    logging_dict['ETH'].append(current_eth_price)
    logging_dict['BTC'].append(current_btc_price)
    
    #fetches the previous z scores
    z_scores = env.get_z_scores('BTCUSDT','ETHUSDT',context_length)['open']
    current_z_score = z_scores[-1]
    
    #add to logging dict
    logging_dict['z_score'].append(current_z_score)
    
    
    #get the cointegration metrics
    coint_p_value, adf1, adf2 = env.calc_coint_values('BTCUSDT','ETHUSDT',100)
    
    #go long on arb is z score is too low
    if current_z_score < -2 and not is_bought:
        env.buy_token('BTCUSDT',10)
        env.short_token('ETHUSDT',10)
        is_bought = True
        logging_dict['action'].append(1)
    #exit position when spread reverts to mean
    elif current_z_score >= 0 and is_bought:
        env.close_all_positions()
        logging_dict['action'].append(1)
        is_bought = False
    else:
        logging_dict['action'].append(0)
        
    #step enviroment time and check if the eipsode is done
    done = env.step_time(1)
        
#cash out any open positions
env.close_all_positions()

#print gain / loss
print(f"loss / gain: {(start_money - env.money)/start_money * 100}%")

#Extract actions
actions = logging_dict.pop('action')
is_buy = True
buy_indices = []
sell_indices = []
for i in range(len(actions)):
    if actions[i] == 1:
        if is_buy:
            buy_indices.append(i)  # It's a buy
        else:
            sell_indices.append(i)  # It's a sell
        is_buy = not is_buy  
        
        
#plot actions
num_plots = len(logging_dict.keys())
fig, axes = plt.subplots(num_plots, 1, figsize=(8, num_plots * 3))

for ax, (key, array) in zip(axes, logging_dict.items()):
    ax.plot(array)  # Line plot
    ax.scatter(buy_indices, [array[i] for i in  buy_indices], marker='^', color='green', label="Buy", s=100, zorder=5)
    ax.scatter(sell_indices, [array[i] for i in sell_indices] , marker='v', color='red', label="Sell", s=100, zorder=5)
    ax.set_title(key)  # Use dictionary key as title
    ax.grid(True)
    
        
    