from envs.binance_trading_enviroment import BinanceTradingEnv
import numpy as np

class StatarbExperienceGenerator:


    def __init__(self,env,token_pair):
        self.env = env
        self.token_pair = token_pair

    def generate_experiences(self,amount):

        assert 'z_score' in self.env.observation_space.keys() and 'is_bought' in self.env.observation_space.keys()

        experiences = []

        obs, _ = self.env.reset()

        action = 0

        for i in range(0,amount):

            current_z_score = obs['z_score'][-1]
            is_bought = obs['is_bought']
            
                #go long on arb is z score is too low
            if current_z_score < -2 and not is_bought:
                self.env.buy_token(self.token_pair[0],1)
                self.env.short_token(self.token_pair[1],1)
                is_bought = True
                action = 1
            #exit position when spread reverts to mean
            elif current_z_score >= 0 and is_bought:
                self.env.close_all_positions()
                action = 1
                is_bought = False
            else:
                action = 0
                
            next_obs, reward, done, truncated, info =  self.env.step(action)
            experiences.append((obs,next_obs,action,reward,done,{}))

            if done:
                obs, _= self.env.reset()
            else:
                obs = next_obs
                
        return experiences
