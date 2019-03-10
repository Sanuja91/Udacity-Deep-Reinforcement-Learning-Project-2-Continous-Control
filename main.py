from agents import Actor_Crtic_Agent
from interact import interact
from utilities import initialize_env, get_device

import numpy as np


env, env_info, state, state_size, action_size, brain_name = initialize_env()
num_agents = len(env_info.agents)
scores = np.zeros(num_agents) 

env_info = env.reset(train_mode = True)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)

agent = Actor_Crtic_Agent(brain_name, get_device(), state_size, num_agents, action_size, load_agent = False)
interact(agent, env, brain_name, n_episodes = 2000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995)

