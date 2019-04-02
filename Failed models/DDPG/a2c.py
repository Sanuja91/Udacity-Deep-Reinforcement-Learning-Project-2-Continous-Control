import numpy as np
import random
import torch
from collections import deque
from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device, update_csv

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 10       
RANDOM_SEED = 4

def actor_critic(agent_name, multiple_agents = False, n_episodes = 300, max_t = 1000):
    """ Batch processed the states in a single forward pass with a single neural network
    Params
    ======
        multiple_agents (boolean): boolean for multiple agents
        PER (boolean): 
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(multiple_agents)
    
    device = get_device()
    scores_window = deque(maxlen=100)
    scores = np.zeros(num_agents)
    scores_episode = []

    max_trajectory_size = max_t // 10
    min_trajectory_size = 1
    agent = Actor_Crtic_Agent(brain_name, agent_name, device, state_size, action_size, num_agents, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED, max_trajectory_size, min_trajectory_size)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        
        agent.reset()
        scores = np.zeros(num_agents)
            
        for t in range(max_t):      
            actions, values, log_probs = agent.act(states)
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done        
            
            if multiple_agents:
                agent.step(states, actions, rewards, next_states, dones, values, log_probs)
            else:
                agent.step(states, np.expand_dims(actions, axis=0), rewards, next_states, dones, values, log_probs)
 
            states = next_states
            scores += rewards
            if t % 20:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="") 
            if np.any(dones):
                break 

        score = np.mean(scores)
        scores_window.append(score)       # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tMax Score: {:.2f}'.format(i_episode, score, np.mean(scores_window), np.max(scores)), end="\n")
        update_csv(agent_name, i_episode, np.mean(scores_window), np.max(scores))
        agent.save_agent(agent_name)

        # Early stop
        if i_episode == 100:
            return scores_episode

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            agent.save_agent(agent_name + "Complete")
            break
            
    return scores_episode
    

