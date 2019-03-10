from unityagents import UnityEnvironment
import numpy as np
import torch

def initialize_env():
    env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe", worker_id = 1, seed = 1)

    """Resetting environment"""
    brain_name = env.brain_names[0]
    print("############## BRAIN NAMES", env.brain_names[0])
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode = True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    # print('States look like:', states[0])
    state_size = len(states[0])

    print('States have length:', state_size)
    print('States initialized:', len(states))

    return env, env_info, states, state_size, action_size, brain_name

# Checks if GPU is available else runs on CPU
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('\nSelected device = {}\n'.format(device))
    return device

# Prints a Break Comment for easy visibility on logs
def print_break(string):
    print("\n################################################################################################ " + string + "\n")

# Return normalized data
def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x





