from collections import deque

import time

import numpy as np

import torch

from utilities import update_csv

NEGATIVE_REWARD = -0.001
PARAMETER_NOISE = 0.01

def train(agents, params, num_processes):
    """Training Loop for value-based RL methods.
    Params
    ======
        agent (object) --- the agent to train
        params (dict) --- the dictionary of parameters
    """
    n_episodes = params['episodes']
    maxlen = params['maxlen']
    name = params['name']
    brain_name = params['brain_name']
    env = params['environment']
    achievement = params['achievement']
    add_noise = params['agent_params']['add_noise']
    pretrain = params['pretrain']
    pretrain_length = params['pretrain_length']
    num_agents = num_processes
    scores = np.zeros(num_agents)                     # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)              # last N scores
    scores_episode = []

    env_info = env.reset(train_mode=True)[brain_name]
    tic = time.time()
    best_min_score = 0.0
    timesteps = 0
    for i_episode in range(1, n_episodes+1):
        timestep = time.time()
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        # agents.reset()
        
        while True:
            states = torch.tensor(states)

            if pretrain and pretrain_length > 0:
                pretrain_length -= 1
            else:
                pretrain = False

            actions = agents.act(states, add_noise, pretrain = pretrain)

            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            adjusted_rewards = np.array(env_info.rewards)

            # adjusted_rewards[adjusted_rewards == 0] = NEGATIVE_REWARD
            # adjusted_rewards = torch.from_numpy(adjusted_rewards).to(device).float().unsqueeze(1)

            agents.step(states, actions, adjusted_rewards, next_states, dones, pretrain = pretrain) 

            scores += rewards                              # update the scores
            states = next_states                           # roll over the state to next time step
            if np.any(dones):                              # exit loop if episode finished
                break
                
            print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'.format(timesteps, np.mean(scores), np.min(scores), np.max(scores)), end="")  
            timesteps += 1 
            
        score = np.mean(scores)
        scores_episode.append(score)
        scores_window.append(score)       # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f} \t Min: {:.2f} \t Max: {:.2f} \t Time: {:.2f}'.format(i_episode, np.mean(scores), np.min(scores), np.max(scores), time.time() - timestep), end="\n")
        if i_episode % 20 == 0:
            agents.save_agent(np.mean(scores_window), i_episode, save_history = True)
        else:
            agents.save_agent(np.mean(scores_window), i_episode, save_history = False)

        update_csv(agents.name, i_episode, np.mean(scores), np.mean(scores))
        if i_episode % 100 == 0:
            toc = time.time()
            print('\rEpisode {}\tAverage Score: {:.2f} \t Min: {:.2f} \t Max: {:.2f} \t Time: {:.2f}'.format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window), toc - tic), end="")
        if np.mean(scores_window) >= achievement:
            toc = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} \t Time: {:.2f}'.format(i_episode-100, np.mean(scores_window), toc-tic))
            if best_min_score < np.min(scores_window):
                best_min_score = np.min(scores_window)
                # agents.save
                # for idx, a in enumerate(agents):
                #     torch.save(a.actor_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_actor_checkpoint.pth')
                #     torch.save(a.critic_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_critic_checkpoint.pth')
    return scores