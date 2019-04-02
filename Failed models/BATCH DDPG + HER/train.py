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
    her = params['agent_params']['her']
    add_noise = params['agent_params']['add_noise']
    num_agents = num_processes
    scores = np.zeros(num_agents)                     # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)              # last N scores
    scores_episode = []
    
    
    # her = HER(self.N)
    # her.reset()

    env_info = env.reset(train_mode=True)[brain_name]
    tic = time.time()
    best_min_score = 0.0
    total_timesteps = 0
    episodes = []
    for i_episode in range(1, n_episodes+1):
        timestep = time.time()
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        # agents.add_param_noise(PARAMETER_NOISE)
        # for a in agents:
        #     a.reset()                                  # reset the noise process after each episode
        agents.reset()
        
        episode_timesteps = 0
        while True:
            
            # actions = [agent.act(states[idx], add_noise) for idx, agent in enumerate(agents)]
            actions = agents.act(states, add_noise)
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            
            adjusted_rewards = np.array(env_info.rewards)
            # adjusted_rewards[adjusted_rewards == 0] = NEGATIVE_REWARD
            # adjusted_rewards = torch.from_numpy(adjusted_rewards).to(device).float().unsqueeze(1)
            
            # TODO Make this happen parallely 


            for idx in range(num_processes):
                # print("\nSTATES\n", states[idx], "\nACTIONS\n", actions[idx], "\nREWARDS\n", rewards[idx], "\nNEXT STATES\n", next_states[idx], "\DONES\n", dones[idx])
                # her.keep([states[idx], actions[idx], adjusted_rewards[idx], next_states[idx], dones[idx]])
                if her:
                    agents.step_her(idx, episode_timesteps, states[idx], actions[idx], adjusted_rewards[idx], next_states[idx], dones[idx]) 
                else:
                    agents.step(states[idx], actions[idx], adjusted_rewards[idx], next_states[idx], dones[idx]) 
                # if dones[idx]:
                #     her.keep([states[idx], actions[idx], adjusted_rewards[idx], next_states[idx], dones[idx]])
            
                agents.learn()

            
            # for a in agents:                               # each agent takes a step, but we give all agents the entire tuple for the experience replay
            #     a.step(states, actions, rewards, next_states, dones)
            # for a in agents:                               # each agent takes a step, but we give all agents the entire tuple for the experience replay
            #     a.learn()
            
            # TODO Make this happen parallely 

            scores += rewards                              # update the scores
            states = next_states                           # roll over the state to next time step
            if np.any(dones):                              # exit loop if episode finished
                # if her:

                break
                
            print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'.format(total_timesteps, np.mean(scores), np.min(scores), np.max(scores)), end="")  
            episode_timesteps += 1
            total_timesteps += 1 

        # her_list = her.backward()     
        # for item in her_list:
        #     print(item)
            # self.replay_buffer.append(item)

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