import numpy as np
import random, torch
from collections import namedtuple, deque

def interact(agent, env, brain_name, n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Agent interacts with the environment
    
    Params
    =========
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
    """
    scores = []                           # list containing scores from each episode
    scores_window = deque(maxlen=100)     # last 100 scores
    eps = eps_start                       # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        
        """Resetting environment"""
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        count = 0
        while True:
            action = agent.act(state, eps).astype(int)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
                  
            agent.step(state, action, reward, next_state, done)    # TODO add priority here
            state = next_state
            score += reward
            count += 1
            if done:
                break
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            agent.save_agent("COMPLETE - " + agent.nn_type)
            # torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores