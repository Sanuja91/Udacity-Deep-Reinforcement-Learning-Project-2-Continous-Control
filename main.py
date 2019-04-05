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
    'no_graphics': True,
    'train_mode': True,
    'buffer_size': 100000,       # size of the replay buffer
    'batch_size': 128,           # batch size sampled from the replay buffer
    'train_batch_size':3000,     # minimum experience required in replay buffer to start training 
    'rollout': 5,                # n step rollout length    
    'agent_count': 20 if MULTI else 1,  
    'gamma': 0.99,
    'device': device
}


env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(environment_params)

seedGenerator = Seeds('seeds')
seedGenerator.next()

experience_params = {
    'seed': seedGenerator,       # seed for the experience replay buffer
    'buffer_size': 100000,       # size of the replay buffer
    'batch_size': 128,           # batch size sampled from the replay buffer
    # 'pretrain': 3000,             # minimum experience required in replay buffer to start training 
    'pretrain': 128,             # minimum experience required in replay buffer to start training 
    'rollout': 10,                # n step rollout length    
    'agent_count': 20 if MULTI else 1,  
    'gamma': 0.99,
    'device': device
}

experienceReplay = NStepReplayBuffer(experience_params)

params = {
    'name': 'D4PG',
    'episodes': 2000,            # number of episodes
    'maxlen': 100,               # sliding window size of recent scores
    'brain_name': brain_name,    # the brain name of the unity environment
    'achievement': 30.,           # score at which the environment is considered solved
    'environment': env,
    'agent_params': {
        'experience_replay': experienceReplay,
        'device': device,
        'seed': seedGenerator,
        'num_agents': num_agents,    # number of agents in the environment
        'gamma': 0.99,               # discount factor
        'tau': 0.001,                # mixing rate soft-update of target parameters
        'update_every': 10,        # update every n-th step
        'update_type': 'hard',      # should the update be soft at every time step or hard at every x timesteps
        'num_updates': 5,            # we don't necessarily need to run as many rounds of updates as there are agents
        'add_noise': True,          # add noise using 'noise_params'
        'actor_params': {            # actor parameters
            'norm': True,
            'lr': 1e-4,            # learning rate
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,                # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            # 'act_fn': [F.leaky_relu, F.leaky_relu, F.F.leaky_relu]
            'act_fn': [nn.ReLU(), nn.ReLU(), nn.Tanh()]
            # nn.ELU()
        },
        'critic_params': {               # critic parameters
            'norm': True,
            'lr': 1e-3,                # learning rate
            'weight_decay': 5e-8,          # weight decay
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,               # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            'action_layer': 1,
            'num_atoms': 51,
            'v_min': 0.0, 
            'v_max': 0.5, 
            # 'act_fn': [F.leaky_relu, F.leaky_relu, lambda x: x]
            'act_fn': [nn.ReLU(), nn.ReLU(), lambda x: x]
        },
        'noise_params': {                  # parameters for the noisy process
            'mu': 0.,                      # mean
            'theta': 0.15,                 # theta value for the ornstein-uhlenbeck process
            'sigma': 0.2,                  # variance
            'seed': seedGenerator,         # seed
            'action_size': action_size
        }
    }
}

# agents = [DDPGAgent(idx=idx, params=params['agent_params']) for idx, a in enumerate(range(num_agents))]
# agents = DDPGAgent(idx=0, params=params['agent_params']) 
agents = D4PGAgent(params=params['agent_params']) 

scores = train(agents=agents, params=params, num_processes=num_agents)

df = pd.DataFrame(data={'episode': np.arange(len(scores)), 'DDPG-3': scores})
df.to_csv('results/DDPG-3-scores.csv', index=False)