import gym
from Agents import QAgent
import numpy as np
from crawler_env import CrawlingRobotEnv

#instantiating crawler environment
env = CrawlingRobotEnv(render=True)

#instantiating agent
agent = QAgent(env, gamma=0.9)

all_rewards = 0

#reset environment and current state, all values are zero
current_state = env.reset()
total_reward = 0
i = 0

while i < 90000: #exploitation and exploration.
    i = i+1

    #choose an action from current state. method from crawler_env
    action = agent.choose_action(current_state)

    #next state and reward values, are obtain from the agents observation
    #depending on the action they've taken.
    next_state, reward, done, info = env.step(action)

    #learn from the taken actions.
    agent.learn(current_state, action, reward, next_state)

    #move on to the next state.
    current_state = next_state

    #add the obtained reward.
    total_reward += reward

    if i % 4000 == 0:

        #print the average reward to evaluate the learning of our agent.
        print("average_reward in last 4000 steps", total_reward / i)

        #after 4000 steps reset the average reward value. 
        #In the next 4000 steps the average reward value 
        #should be higher than the one in the previous 
        # 4000 steps. 
        average_reward = 0
        env.render = True

env = CrawlingRobotEnv(render=True)

#reset the environment a second time for exploitation.
current_state = env.reset()
total_reward = 0

#don't explore, only exploit.
agent.eps = 0

i = 0
while True: #for exploitation. agent should be very fast.
    i = i+1
    action = agent.choose_action(current_state)
    next_state, reward, done, info = env.step(action)
    agent.learn(current_state, action, reward, next_state)
    current_state = next_state
    total_reward += reward

    if i % 5000 == 0:  #evaluation
        
        #break the loop after some time
        if total_reward/i > 9:
            break

        print("average_reward in last 5000 steps", total_reward / 5000)
        i = 0
    average_reward = 0
    env.render = True
