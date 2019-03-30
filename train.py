from collections import deque

import time

import numpy as np

import torch

def train(agents, params):
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
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=maxlen)  # last N scores

    env_info = env.reset(train_mode=True)[brain_name]
    tic = time.time()
    best_min_score = 0.0
    for i_episode in range(1, n_episodes+1):
        timestep = time.time()
        states = env_info.vector_observations
        score = 0.0
        for a in agents:
            a.reset()                                  # reset the noise process after each episode
        while True:
            actions = [agent.act(states[idx], add_noise) for idx, agent in enumerate(agents)]
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done                    # see if episode has finished
            for a in agents:                               # each agent takes a step, but we give all agents the entire tuple for the experience replay
                a.step(states, actions, rewards, next_states, dones)
            for a in agents:                               # each agent takes a step, but we give all agents the entire tuple for the experience replay
                a.learn()
            score += np.mean(rewards)                              # update the score
            states = next_states                           # roll over the state to next time step
            if np.any(dones):                              # exit loop if episode finished
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f} \t Min: {:.2f} \t Max: {:.2f} \t Time: {:.2f}'.format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window), time.time() - timestep), end="")
        if i_episode % 100 == 0:
            toc = time.time()
            print('\rEpisode {}\tAverage Score: {:.2f} \t Min: {:.2f} \t Max: {:.2f} \t Time: {:.2f}'.format(i_episode, np.mean(scores_window), np.min(scores_window), np.max(scores_window), toc - tic), end="")
        if np.mean(scores_window) >= achievement:
            toc = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} \t Time: {:.2f}'.format(i_episode-100, np.mean(scores_window), toc-tic))
            if best_min_score < np.min(scores_window):
                best_min_score = np.min(scores_window)
                for idx, a in enumerate(agents):
                    torch.save(a.actor_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_actor_checkpoint.pth')
                    torch.save(a.critic_local.state_dict(), 'results/' + str(idx) + '_' + str(i_episode) + '_' + name + '_critic_checkpoint.pth')
    return scores