from unityagents import UnityEnvironment
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

from agent import DDPGAgent
from train import train
from utilities import Seeds, initialize_env
from memory import UniformReplayBuffer, HindsightReplayBuffer

HER = False
MULTI = True

env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(multiple_agents = MULTI, train_mode = True)

seedGenerator = Seeds('seeds')
seedGenerator.next()

if HER:
    # num_agents = 
    experience_params = {
    'seed': seedGenerator,       # seed for the experience replay buffer
    'buffer_size': 100000,        # size of the replay buffer
    'batch_size': 128,            # batch size sampled from the replay buffer
    'num_agents': 20 if MULTI else 1,
    'timesteps_per_episode': 1000
    }
    experienceReplay = HindsightReplayBuffer(experience_params)

else:

    experience_params = {
        'seed': seedGenerator,       # seed for the experience replay buffer
        'buffer_size': 100000,        # size of the replay buffer
        'batch_size': 128            # batch size sampled from the replay buffer
    }

    experienceReplay = UniformReplayBuffer(experience_params)

params = {
    'name': 'DDPG-3',
    'episodes': 2000,            # number of episodes
    'maxlen': 100,               # sliding window size of recent scores
    'brain_name': brain_name,    # the brain name of the unity environment
    'achievement': 30.,           # score at which the environment is considered solved
    'environment': env,
    'agent_params': {
        'experience_replay': experienceReplay,
        'her': HER,
        'seed': seedGenerator,
        'num_agents': num_agents,    # number of agents in the environment
        'gamma': 0.99,               # discount factor
        'tau': 0.001,                # mixing rate soft-update of target parameters
        'update_every': 10,        # update every n-th step
        'num_updates': 5,            # we don't necessarily need to run as many rounds of updates as there are agents
        'add_noise': True,          # add noise using 'noise_params'
        'actor_params': {            # actor parameters
            'norm': False,
            'lr': 1e-4,            # learning rate
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,                # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            # 'act_fn': [F.leaky_relu, F.leaky_relu, F.F.leaky_relu]
            'act_fn': [nn.ELU(), nn.ELU(), nn.ELU()]
            # nn.ELU()
        },
        'critic_params': {               # critic parameters
            'norm': False,
            'lr': 1e-4,                 # learning rate
            'weight_decay': 0.0,          # weight decay
            'state_size': state_size,    # size of the state space
            'action_size': action_size,  # size of the action space
            'seed': seedGenerator,               # seed of the network architecture
            'hidden_layers': [512, 512, 128], # hidden layer neurons
            'dropout': 0.05,
            'action_layer': 1,
            # 'act_fn': [F.leaky_relu, F.leaky_relu, lambda x: x]
            'act_fn': [nn.ELU(), nn.ELU(), lambda x: x]
        },
        'noise_params': {            # parameters for the noisy process
            'mu': 0.,                # mean
            'theta': 0.15,           # theta value for the ornstein-uhlenbeck process
            'sigma': 0.2,            # variance
            'seed': seedGenerator,         # seed
            'action_size': action_size
        }
    }
}

# agents = [DDPGAgent(idx=idx, params=params['agent_params']) for idx, a in enumerate(range(num_agents))]
agents = DDPGAgent(idx=0, params=params['agent_params']) 

scores = train(agents=agents, params=params, num_processes=num_agents)

df = pd.DataFrame(data={'episode': np.arange(len(scores)), 'DDPG-3': scores})
df.to_csv('results/DDPG-3-scores.csv', index=False)