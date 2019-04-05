from unityagents import UnityEnvironment
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

from agent import DDPGAgent, D4PGAgent
from train import train
from utilities import Seeds, initialize_env, get_device
from memory import NStepReplayBuffer

MULTI = True
device = get_device()

environment_params = {
    'multiple_agents': True,      
    'no_graphics': False,
    'train_mode': True,
    'agent_count': 20 if MULTI else 1,  
    'device': device
}


env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(environment_params)

seedGenerator = Seeds('seeds')
seedGenerator.next()

experience_params = {
    'seed': seedGenerator,       # seed for the experience replay buffer
    'buffer_size': 100000,       # size of the replay buffer
    'batch_size': 128,           # batch size sampled from the replay buffer
    'rollout': 5,                # n step rollout length    
    'agent_count': 20 if MULTI else 1,  
    'gamma': 0.99,
    'device': device
}

experienceReplay = NStepReplayBuffer(experience_params)

params = {
    'episodes': 2000,            # number of episodes
    'maxlen': 100,               # sliding window size of recent scores
    'brain_name': brain_name,    # the brain name of the unity environment
    'achievement': 30.,           # score at which the environment is considered solved
    'environment': env,
    'pretrain': True,            # whether pretraining with random actions should be done
    'pretrain_length': 3000,     # minimum experience required in replay buffer to start training 
    'random_fill': True,         # basically repeat pretrain at specific times to encourage further exploration
    'random_fill_every': 10000,
    'log_dir': 'runs/',
    'agent_params': {
        'name': 'D4PG',
        'load_agent': True,
        'experience_replay': experienceReplay,
        'device': device,
        'seed': seedGenerator,
        'num_agents': num_agents,    # number of agents in the environment
        'gamma': 0.99,               # discount factor
        'tau': 0.001,                # mixing rate soft-update of target parameters
        'update_every': 300,        # update every n-th step
        'update_type': 'hard',      # should the update be soft at every time step or hard at every x timesteps
        'add_noise': True,          # add noise using 'noise_params'
        'actor_params': {            # actor parameters
            'norm': True,
            'lr': 0.0005,            # learning rate
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,                # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            'act_fn': [nn.ReLU(), nn.ReLU(), nn.Tanh()]
        },
        'critic_params': {               # critic parameters
            'norm': True,
            'lr': 0.001,                # learning rate
            'weight_decay': 0.0001,          # weight decay
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,               # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            'action_layer': True,
            'num_atoms': 51,
            'v_min': 0.0, 
            'v_max': 0.5, 
            # 'act_fn': [F.leaky_relu, F.leaky_relu, lambda x: x]
            'act_fn': [nn.ReLU(), nn.ReLU(), lambda x: x]
        },
        'ou_noise_params': {               # parameters for the Ornstein Uhlenbeck process
            'mu': 0.,                      # mean
            'theta': 0.15,                 # theta value for the ornstein-uhlenbeck process
            'sigma': 0.2,                  # variance
            'seed': seedGenerator,         # seed
            'action_size': action_size
        },
        'ge_noise_params': {               # parameters for the Gaussian Exploration process                  
            'max_epsilon': 1,                 
            'min_epsilon': 0.05,         
            'decay_rate': 0.99998
        },
        
    }
}


agents = D4PGAgent(params=params['agent_params']) 

scores = train(agents=agents, params=params, num_processes=num_agents)

df = pd.DataFrame(data={'episode': np.arange(len(scores)), 'DDPG-3': scores})
df.to_csv('results/DDPG-3-scores.csv', index=False)