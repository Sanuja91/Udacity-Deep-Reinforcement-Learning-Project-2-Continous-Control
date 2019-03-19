import numpy as np
import random
import torch
from collections import deque
from memory import NaivePrioritizedBuffer, ReplayBuffer
from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128       
RANDOM_SEED = 4

def ensure_shared_grads(model, shared_model):
    """"""
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def share_learning(shared_model, agents):
    """Distribute the learning across all agents"""
    new_agents = []
    for agent in agents:
        ensure_shared_grads(agent.actor_local, shared_model)
        new_agents.append(agent)

    return new_agents


def ddpg(multiple_agents = False, PER = False, n_episodes = 300, max_t = 1000):
    """ Deep Deterministic Policy Gradients
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
    
    agents = [] 
    shared_memory = ReplayBuffer(device, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
    for agent_id in range(num_agents):
        agents.append(Actor_Crtic_Agent(brain_name, agent_id, device, state_size, action_size))
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        
        for agent in agents:
            agent.reset()
            
        scores = np.zeros(num_agents)
            
        for t in range(max_t):      
            actions = np.array([agents[i].act(states[i]) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done        
            
            for i in range(num_agents):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], shared_memory) 
            if shared_memory.batch_passed():
                experiences = shared_memory.sample()
                agents[0].learn(experiences, shared_memory)
                agents = share_learning(agents[0].actor_local, agents)
 
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

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)), end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores_episode


def batch_ddpg(multiple_agents = False, PER = False, n_episodes = 300, max_t = 1000):
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
    
    agents = [] 
    shared_memory = ReplayBuffer(device, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
    agent = Actor_Crtic_Agent(brain_name, 0, device, state_size, action_size)
    # for agent_id in range(num_agents):
    #     agents.append(Actor_Crtic_Agent(brain_name, agent_id, device, state_size, action_size))
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        
        agent.reset()
        # for agent in agents:
        #     agent.reset()
            
        scores = np.zeros(num_agents)
            
        for t in range(max_t):      
            actions = agent.act(states)
            # actions = np.array([agent.act(states[i]) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done        
            
            
            agent.step(states, actions, rewards, next_states, dones, shared_memory)
            # for i in range(num_agents):
            #     agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], shared_memory) 
            if shared_memory.batch_passed():
                experiences = shared_memory.sample()
                agent.learn(experiences, shared_memory)
                # agents = share_learning(agents[0].actor_local, agents)
 
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

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)), end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores_episode