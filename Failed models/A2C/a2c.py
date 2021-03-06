import numpy as np
import random
import torch
import time
from collections import deque
from models import ActorCritic
from agents import A2C_ACKTR
from storage import SimpleRolloutStorage
from utilities import initialize_env, get_device, update_csv

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 10       
RANDOM_SEED = 4
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = 0.95
ENTROPY_BETA = 0.01
CRITIC_DISCOUNT = 0.5
EPS = 1e-5
ALPHA = 0.99
MAX_GRAD_NORM = 5
NUM_PROCESSES = 20
NUM_ENV_STEPS = 10000000
NUM_STEPS = 5
# NUM_STEPS = 30
PPO_EPOCH = 4
PPO_CLIP = 0.2
NUM_MINI_BATCH = 32
GAMMA = 0.99
USE_GAE = True
GAE_LAMBDA = 0.95
MAX_EPISODES = 2000
NEGATIVE_REWARD = -0.001
ACTION_BOUNDS = [-1, 1]
PARAMETER_NOISE = 0.01

def actor_critic(agent_name, multiple_agents = False, load_agent = False, n_episodes = 300, max_t = 1000, train_mode = True):
    """ Batch processed the states in a single forward pass with a single neural network
    Params
    ======
        multiple_agents (boolean): boolean for multiple agents
        PER (boolean): 
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    start = time.time()
    device = get_device()
    env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(multiple_agents, train_mode)
    states = torch.from_numpy(states).to(device).float()
    
    NUM_PROCESSES = num_agents

    # Scores is Episode Rewards
    scores = np.zeros(num_agents)
    scores_window = deque(maxlen=100)
    scores_episode = []
    
    actor_critic = ActorCritic(state_size, action_size, device).to(device)
    agent = A2C_ACKTR(agent_name, actor_critic, value_loss_coef = CRITIC_DISCOUNT, entropy_coef = ENTROPY_BETA, lr = LEARNING_RATE, eps = EPS, alpha = ALPHA, max_grad_norm = MAX_GRAD_NORM, acktr = False, load_agent = load_agent)
    
    rollouts = SimpleRolloutStorage(NUM_STEPS, NUM_PROCESSES, state_size, action_size)
    rollouts.to(device)
    
    num_updates = NUM_ENV_STEPS // NUM_STEPS // NUM_PROCESSES
    # num_updates = NUM_ENV_STEPS // NUM_STEPS

    print("\n## Loaded environment and agent in {} seconds ##\n".format(round((time.time() - start), 2)))
    
    update_start = time.time()
    timesteps = 0
    episode = 0
    if load_agent != False:
        episode = agent.episode
    while True:   
        """CAN INSERT LR DECAY HERE"""
        # if episode == MAX_EPISODES:
        #     return scores_episode

        # Adds noise to agents parameters to encourage exploration
        # agent.add_noise(PARAMETER_NOISE)

        for step in range(NUM_STEPS):
            step_start = time.time()

            # Sample actions
            with torch.no_grad():
                values, actions, action_log_probs, _  = agent.act(states)


            clipped_actions = np.clip(actions.cpu().numpy(), *ACTION_BOUNDS)
            env_info = env.step(actions.cpu().numpy())[brain_name]       # send the action to the environment 
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            rewards_tensor = np.array(env_info.rewards)
            rewards_tensor[rewards_tensor == 0] = NEGATIVE_REWARD
            rewards_tensor = torch.from_numpy(rewards_tensor).to(device).float().unsqueeze(1)
            dones = env_info.local_done  
            masks = torch.from_numpy(1 - np.array(dones).astype(int)).to(device).float().unsqueeze(1) 

            rollouts.insert(states, actions, action_log_probs, values, rewards_tensor, masks, masks)
            
            next_states = torch.from_numpy(next_states).to(device).float()
            states = next_states
            scores += rewards
            # print(rewards)

            if timesteps % 100:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'.format(timesteps, np.mean(scores), np.min(scores), np.max(scores)), end="") 
            

            
            if np.any(dones):
                print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tMin Score: {:.2f}\tMax Score: {:.2f}'.format(episode, score, np.mean(scores_window), np.min(scores), np.max(scores)), end="\n")
                update_csv(agent_name, episode, np.mean(scores_window), np.max(scores))

                if episode % 20 == 0:
                    agent.save_agent(agent_name, score, episode, save_history = True)
                else:
                    agent.save_agent(agent_name, score, episode)

                episode += 1
                scores = np.zeros(num_agents)
                break 

            timesteps += 1

        
        with torch.no_grad():
            next_values, _, _, _  = agent.act(next_states)
        
        rollouts.compute_returns(next_values, USE_GAE, GAMMA, GAE_LAMBDA)
        agent.update(rollouts)
    
        score = np.mean(scores)
        scores_window.append(score)       # save most recent score
        scores_episode.append(score)

    return scores_episode