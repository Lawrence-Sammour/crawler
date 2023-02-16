import random
from collections import defaultdict
import numpy as np

#The class body that ill be used to instantiate an agent in play.py
class QAgent():

    #Constructor that takes the initialized environment and the discount rate as parameters
    def __init__(self,env,gamma):
        
        self.gamma = 0.9
        self.env = env

        #contains the value of state numbers
        self.q_vals = defaultdict(lambda: np.array([0. for _ in range(env.action_space.n)]))
        self.alpha = 0.2

        #balance between exploration and exploitation.
        self.eps = 0.5

    #choose an action function takes the state that the agent is in to
    def choose_action(self,state):
        #if the the generated random number is less than 0.5 eps explore a new path
        if random.random() < self.eps:
            action = np.random.choice(len(self.q_vals[state]))
        else:
            #if the random generated number is larger than eps exploit the explored path
            action = np.argmax(self.q_vals[state])
        return action

    #learn function takes the current state, reward, next state
    #oldv is calculated backwards depending on new value.
    #new value is the new state when the agent moves forward.
    def learn(self, cur_state,action,reward,next_state):

        #takes the maximum value of state between all possible actions of the 
        #prime state to calculate the current state since we're calculating backwards
        #maxqnew = Q(s,a) prime
        maxqnew = np.max(self.q_vals[next_state])

        #adds the reward that is obtained from the 
        #taken action and multiplying V(s) with discount
        new_value = reward + self.gamma*maxqnew

        #getting the values of state for the current state that we want to calculate
        #which is the old state when the agent is moving forward
        #oldv = Q(s,a) current
        oldv = self.q_vals[cur_state][action]

        #updating action function each time by adding the old value
        self.q_vals[cur_state][action] = oldv + self.alpha * (new_value - oldv)