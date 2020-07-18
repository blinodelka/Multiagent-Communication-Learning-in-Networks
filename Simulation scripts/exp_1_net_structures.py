#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:20:55 2020

@author: marinadubova
"""


import numpy as np
from networkx import nx

import sys
if sys.version[0] == '3':
    import pickle
else:
    import cPickle as pickle
import collections
import random


import keras 

from keras.layers import Lambda, Input, Dense, LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.regularizers import l1, l2
from keras.utils import np_utils, plot_model
from keras.layers import concatenate
from keras.layers.core import Reshape
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os

from keras import backend as K

import sys

trial_index = int(sys.argv[1])

print("trial index:" + str(trial_index))
class Two_Roles_Game_Many_Agents():

    '''Environment for the game. Keeps track of action and message use, and administers rewards'''

    def __init__(
        self, num_choices, winning_reward, mean_sample, talking_action_size, num_agents, mean_punishment_weight_talk, punishment_weight):
        self.num_choices = num_choices
        self.talking_action_size = talking_action_size
        self.winning_reward = winning_reward
        self.mean_sample = mean_sample
        previous_actions = []
        previous_talks = []
        self.mean_punishment_weight_talk = mean_punishment_weight_talk
        for i in range(num_agents):
            previous_actions.append(collections.deque(maxlen=mean_sample)) # for mean calculations
        for i in range(num_agents):
            previous_talks.append(collections.deque(maxlen=mean_sample))
        self.previous_talks = previous_talks
        self.previous_actions = previous_actions
        self.mean_punishment_weight = punishment_weight
                 
    def step(self, ag1_action, ag2_action, choose, sample, reward1 = 0, reward2 = 0):
       
        talker_input = np.random.random((self.talking_action_size+self.num_choices+1,)) #UNIFORM/bimodal NOISE + signal for the game role
        talker_input[0] = 1 # indicates that the agent needs to talk
        
        hearer_input = np.zeros((self.talking_action_size+self.num_choices+1,))
        
        # Add information about action use in the past to the input
        for i in range(self.num_choices):
            talker_input[self.talking_action_size+1+i] = self.previous_actions[sample[0]].count(i)/self.mean_sample
            hearer_input[self.talking_action_size+1+i] = self.previous_actions[sample[1]].count(i)/self.mean_sample
        
        if choose: # final step: comparing actions and administering reward
            self.previous_actions[sample[0]].append(ag1_action)
            self.previous_actions[sample[1]].append(ag2_action)
            a1_rew = 0
            a2_rew = 0
            
            if len(self.previous_actions[sample[0]]) >= 2 and len(self.previous_actions[sample[1]]) >= 2:
                
                a1_rew = ((float(1)/self.num_choices) - float(self.previous_actions[sample[0]].count(ag1_action))/len(self.previous_actions[sample[0]])) * self.mean_punishment_weight
                a2_rew = ((float(1)/self.num_choices) - float(self.previous_actions[sample[1]].count(ag2_action))/len(self.previous_actions[sample[1]])) * self.mean_punishment_weight
                if a1_rew>0:
                    a1_rew = 0
                if a2_rew>0:
                    a2_rew = 0
               

            if ag2_action == ag1_action: 
                reward1 = self.winning_reward + a1_rew
                reward2 = self.winning_reward + a2_rew
                
            else:
                reward1 = a1_rew
                reward2 = a2_rew
                                    
        else:       
            talker_action = ag1_action # communication step
            self.previous_talks[sample[0]].append(ag1_action)
            hearer_input[talker_action+1] = 1
            if len(self.previous_talks[sample[0]]) >= 2:
                reward1 = ((float(1)/self.talking_action_size) - float(self.previous_talks[sample[0]].count(ag1_action))/len(self.previous_talks[sample[0]])) * self.mean_punishment_weight_talk
                if reward1 > 0:
                    reward1 = 0
        
        return(talker_input, hearer_input, reward1, reward2)


class DQNAgent_student_teacher:
    def __init__(self, talking_action_size, choices, beta, memory_size = 10): 
        
        # Simulation parameters

        self.memory = collections.deque(maxlen=memory_size)
        self.epsilon = 1.0 # initial exploration rate
        self.epsilon_min = 0.1 # minimal exploration rate
        self.epsilon_decay = 0.999 # exploration rate decay

        self.talking_action_size = talking_action_size # number of available messages
        self.choices = choices # number of available actions
        
        # Optimizer parameters        
        self.beta_1 = beta
        self.learning_rate = 0.0001


        self.model = self._build_model_student_teacher()
 
 
    def _build_model_student_teacher(self):
        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        model.add(Dense(15, input_dim=self.talking_action_size + self.choices + self.talking_action_size+ 1, activation='relu')) # one input - the game role
        model.add(Dense(25, activation='relu'))
 
        model.add(Dense(self.choices, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=0.99))
        return model
       
   
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
                           
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.talking_action_size), random.randrange(self.choices)]
 
        
        max_q = -float("infinity")
        max_talk = None
        max_action = None
 
        for i in range(self.talking_action_size):
            talking = np.zeros(self.talking_action_size)
            talking[i] = 1
 
            state_cur = np.expand_dims(np.hstack([state, talking]), axis=0)
            act_values = self.model.predict(state_cur, batch_size=1)
 
            cur_max = np.max(act_values)
           
            if cur_max > max_q:
 
              max_q = cur_max
              max_action = np.argmax(act_values[0])
              max_talk = i
 
            assert max_talk is not None
            assert max_action is not None
 
        return [max_talk, max_action]
                           
    def replay(self, batch_size):
        for i in self.memory:
            minibatch = random.sample(self.memory, batch_size)
        my_x = []
        my_y = []
        for state, action, reward, next_state in minibatch:
            next_state = np.expand_dims(next_state, axis=0)
            target = reward
            one_hot_talk = np.zeros(self.talking_action_size)
            one_hot_talk[action[0]] = 1
            cur_state = np.expand_dims(np.hstack([state, one_hot_talk]), axis=0)
            target_f = self.model.predict(cur_state, batch_size=1)
            target_f[0][action[1]] = target # choose
            cur_state = np.squeeze(cur_state)
            target_f = np.squeeze(target_f)
            my_x.append(cur_state)
            my_y.append(target_f)
        
        my_x = np.array(my_x)
        my_y = np.array(my_y)
        self.model.train_on_batch(my_x, my_y)
 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def play_many_games_semisupervised(num_agents, num_episodes, inner_speech, learning_rate, punishment_weight, 
    punishment_talk, agents_memory, replay, beta_1, p_peek, num_choose_act, network_type, verbose=False):

    '''Run the game for many episodes to obtain simulation results'''

    # Parameters fixed throughout simulations

    num_choices = num_choose_act
    num_talking_symbols = num_choose_act
    winning_reward = 1
    mean_sample = 100
    punishment_weight = punishment_weight
    num_agents = num_agents
    n_type = network_type
    
    # Initialize environment and agents
    env = Two_Roles_Game_Many_Agents(num_choices, winning_reward, mean_sample, num_talking_symbols, num_agents, punishment_talk, punishment_weight) 

    agents = []
    scores = []
    talks = []
    acts = []
    samples = []

    m = 20
    for i in range(num_agents):
            x = DQNAgent_student_teacher(num_talking_symbols, num_choices, beta_1, agents_memory)
            x.learning_rate = learning_rate
            agents.append(x)
            talks.append([])
            acts.append([])
            scores.append([])
    if n_type==0: #random
        G = nx.gnm_random_graph(num_agents, m)
    elif n_type==1: #fully connected
        G = nx.complete_graph(num_agents)
    elif n_type==2: #small-world
        G = nx.connected_watts_strogatz_graph(num_agents, 2, 0.2) 
    elif n_type==3: #ring
        G = nx.connected_watts_strogatz_graph(num_agents, 2, 0)
            
    # Iterate the game
    episodes = num_episodes
    
    for e in range(episodes):
            
        # Selecting agents to play in the spisode
        my_sample1 = random.sample(range(num_agents), 1)[0]
        my_sample2 = random.choice(list(G.neighbors(my_sample1)))
        my_sample = [my_sample1, my_sample2]

        agent1 = agents[my_sample[0]] # agent1 is always the speaker
        agent2 = agents[my_sample[1]] # agent2 is always the listener

        # Initialize the scores
        score1 = 0
        score2 = 0

        state1, state2, _, _ = env.step(0, 0, 0, my_sample) # initialize the environment

        # agent 1 talks
        action1 = agent1.act(state1)


        # update environment based on agent1's speech and actions
        state1, state2, reward1, reward2 = env.step(action1[0], 0, 0, my_sample)     
        score1 += reward1


        # Agent 2 acts based on agent 1's message
        action2 = agent2.act(state2)

        # Update the environment and coompute coordination rewards
        next_state1, next_state2, reward1, reward2 = env.step(action1[1], action2[1], 1, my_sample)     

        score1 += reward1
        score2 += reward2

        # Save the transition to memory

        agents[my_sample[1]].remember(state2, action2, reward2, next_state2)

        # Semi-supervised updates:
        
        if random.random() < p_peek: # peek another's action 20% of times: mimiking 
            agents[my_sample[1]].remember(state2, action1, winning_reward, next_state2)
            
        agents[my_sample[0]].remember(state1, action1, reward1, next_state1)
        if random.random() < p_peek: 
            agents[my_sample[0]].remember(state1, action2, winning_reward, next_state1)
            
        # Monitor progress
        if e %100 == 0 and verbose:
           print("episode: {}/{}, score1: {}, score2: {}"
                         .format(e, episodes, score1, score2))

        # Train agents
        if len(agent1.memory) >= replay and len(agent2.memory) >= replay: 
            agents[my_sample[0]].replay(replay)
            agents[my_sample[1]].replay(replay)
        else:
            print("replay more than memory")
		
        # Save data for later analysis
        talks[my_sample[0]].append(action1[0])
        talks[my_sample[1]].append(-1) #didn't talk
        acts[my_sample[0]].append(action1[1])
        acts[my_sample[1]].append(action2[1])
        scores[my_sample[0]].append(score1)
        scores[my_sample[1]].append(score2)
        samples.append(my_sample)
        
    return [talks, acts, scores, samples]

cond_list = []
supervision = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9]
samples = 10
n_types = [0,1,2,3]

for superv in supervision:
    for n_type in n_types:
        for j in range(samples):
            cond_list.append([superv, n_type])
            
supervision_rate = cond_list[trial_index][0]
n_type = cond_list[trial_index][1]
d = play_many_games_semisupervised(10, 120000, 0, 0.0001, 4, 0, 10, 10, 0.3, supervision_rate, 4, n_type)
d.append(cond_list[trial_index])


with open("lang_games/game{}.pkl".format(trial_index), "wb") as fp:   #Pickling
    pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
