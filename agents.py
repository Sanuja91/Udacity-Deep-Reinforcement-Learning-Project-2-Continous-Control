import argparse
import math
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Normal

from models import ActorCritic

HIDDEN_SIZE         = 50
LEARNING_RATE       = 0.1e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
PPO_EPSILON         = 0.2
CRITIC_DISCOUNT     = 0.5
ENTROPY_BETA        = 0.001
PPO_STEPS           = 256
MINI_BATCH_SIZE     = 64
PPO_EPOCHS          = 3
TEST_EPOCHS         = 5
NUM_TESTS           = 2
TARGET_REWARD       = 2500
GRADIENT_CLIP       = 5
NUM_TEST_TRAJECTORY_STARTS = 5
LAST_UPDATE_LIMIT = 10000
LEARNING_RATE_DECAY = 0.33
MINIMUM_LEARNING_BUFFER_SIZE = 10

class Actor_Crtic_Agent():
    """Interacts with and learns from the environment"""
    
    def __init__(self, state_size, action_size, load_agent = False, trajectory_length = 48, mini_batch_size = 4, random_seed = 2, adam_eps = 1e-5, buffer_size = 100):
        """Intialize an Agent object
        Params
        =======
        state_size(int): dimension of each state
        action_size(int): dimension of each action
        trajectory_length(int): timesteps per trajectory
        """
        
        self.name = "Actor Critic Catalyst"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Environment details
        self.state_size = state_size
        self.action_size = action_size
        self.trajectory_length = trajectory_length
        self.mini_batch_size = mini_batch_size
        self.test_action_sample = torch.from_numpy(np.ndarray.flatten(np.random.dirichlet(np.ones(self.action_size), size = 1))).float().to(self.device).unsqueeze(0)      # Generate random starting weights
        self.train_trajectory_starts = []
        self.test_trajectory_starts = []       # Keep constant to properly evaluate performance
        self.last_upgraded_frame = 0           # How many frames ago did the agent perform better than the previous best
        self.adam_eps = adam_eps
        self.learning_rate = LEARNING_RATE
        # Neural network details
        self.model = ActorCritic(state_size, action_size, HIDDEN_SIZE, self.mini_batch_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, eps = self.adam_eps)
        self.best_reward = 0
        self.buffer = ReplayBuffer(self.action_size, buffer_size, self.mini_batch_size, self.device)

        if(load_agent):
            self.load_agent(self.name)
 
        print("########################################## NEURAL NETWORKS\n\n")
        print(self.model)
        print("\n\n")

    def step(self, frame_idx, state, next_state, portfolio_weights, final_log_return, done, test_toggle = False):
        """Suggests action depending on environment
        Params
        =======
        state(np array): state
        next_state(np array): next_state
        portfolio_weights(np array): previous action / existing portfolio weights
        final_log_return(np array): the log return for the last step, differnece between state and next state
        done(int): whether trajectory is complete or not
        test_toggle(boolean): toggles between train and test mode
        """

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)

        prev_action = torch.from_numpy(portfolio_weights).float().to(self.device)      # generate random starting weights

        state = state.permute(0, 2, 1)                                                 # transforms the state in the required tensor shape
        next_state = next_state.permute(0, 2, 1)                                       # transforms the state in the required tensor shape

        if test_toggle:
            self.model.eval()

        dist, value, action_sample = self.model(state, prev_action)
        action = self.model.get_action(action_sample)
        mini_rewards = final_log_return * action
        reward = mini_rewards.sum()
        print(frame_idx, (np.exp(reward) - 1) * 100, "%")
        log_probs = dist.log_prob(action_sample)
        self.buffer.add_sub(state, next_state, prev_action, action, log_probs, value, mini_rewards, done)

        if done == 1 and test_toggle == False:
            _, next_value, _ = self.model(next_state, torch.from_numpy(action).float().to(self.device))
            returns = self._compute_gae(next_value, self.buffer.sub_rewards, self.buffer.sub_masks, self.buffer.sub_values)

            returns = torch.cat(returns).detach()
            log_probs = np.concatenate(self.buffer.sub_log_probs)
            values = torch.from_numpy(np.concatenate(self.buffer.sub_values)).float().to(self.device)
            states = np.concatenate(self.buffer.sub_states)
            prev_actions = np.concatenate(self.buffer.sub_prev_actions)
            actions = np.concatenate(self.buffer.sub_actions)
            advantage = returns - values
            self.buffer.add(states, prev_actions, actions, log_probs, values, returns, advantage) 
            
            # print("REPLAY SIZE", self.buffer.__len__())
            if self.buffer.__len__() > MINIMUM_LEARNING_BUFFER_SIZE and test_toggle == False:
                self._learn()

        if test_toggle:
            self.model.train()
        
        return reward
     
    def _compute_gae(self, next_value, rewards, masks, values, gamma = GAMMA, tau = GAE_LAMBDA):
        """Compute Generalized Advantage Estimation"""
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            rewards[step] = torch.from_numpy(rewards[step]).float().to(self.device)
            values[step] = torch.from_numpy(values[step]).float().to(self.device)
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

   
    def _learn(self, clip_param = PPO_EPSILON):
        """Learn from experiences"""
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0

        batch_loss = []
        states, prev_actions, actions, old_log_probs, values, returns, advantage = self.buffer.sample()

        # print("################# LEARNING")
        # PPO EPOCHS is the number of times we will go through ALL the training data to make updates
        for e in range(PPO_EPOCHS):
            # grabs random sample from replay buffer
            dist, new_value, new_action_sample = self.model(states, prev_actions, debug = False)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(new_action_sample)
            ratio = torch.exp(new_log_probs - old_log_probs)
            # print("############## IDX {} NEW ACTION SAMPLE {}".format(idx, new_action_sample))
            actor_surr1 = ratio * advantage
            actor_surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(actor_surr1, actor_surr2).mean()
            # critic_loss = (return_ - value).pow(2).mean()   ## PREVIOUS CODE DON'T DELETE
            ### NEW CODE
            value_clipped = values + torch.clamp(new_value - values, -PPO_EPSILON, PPO_EPSILON)
            critic_surr1 = (values - returns).pow(2)
            critic_surr2 = (value_clipped - returns).pow(2)
            critic_loss = torch.max(critic_surr1, critic_surr2).mean()
            ## NEW CODE 
            loss = actor_loss - ENTROPY_BETA * entropy + CRITIC_DISCOUNT * critic_loss
            batch_loss.append(loss)
            # print("####### LOSS", loss, "CRITIC LOSS", critic_loss, "ACTOR LOSS", actor_loss, "ENTROPY", entropy)
            # print("####### LOSS", loss)
            # print("#### NEW PROBS {} OLD PROBS {} ENTROPHY {}".format(new_log_probs, old_log_probs, entropy))
            # print("############################################################################################## LEARNING\n\n ")
            # print("EPOCH {} IDX {} NEW ACTION SAMPLE {} PREV ACTION SAMPLE {} NEW PROBS {} OLD PROBS {} RATIO {} ADVANTAGE {}\n\n".format(e, idx, new_action_sample, prev_action, new_log_probs, old_log_probs, ratio, advantage))
            # print("ACTOR ## SURR1 {} SURR2 {} A-LOSS {} \n\n".format(actor_surr1, actor_surr2, actor_loss))
            # print("CRITIC ## SURR1 {} SURR2 {} C-LOSS {} \n\n".format(critic_surr1, critic_surr2, critic_loss))        
            # print("EPOCH {} IDX {}".format(e, idx))
         
            self.optimizer.zero_grad()
            # loss /= 1000
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in resulting in NaNs
            nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)
            # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
            
            self.optimizer.step()
            # track statistics
            sum_returns += returns.mean()
            sum_advantage += advantage.mean()
            sum_loss_actor += actor_loss
            sum_loss_critic += critic_loss
            sum_loss_total += loss
            sum_entropy += entropy
            count_steps += 1
            # self.model.train()

    def save_agent(self, fileName):
        """Save the checkpoint"""
        checkpoint = {'state_dict': self.model.state_dict(), 'best_reward': self.best_reward, 'last_upgraded_frame': self.last_upgraded_frame, 'learning_rate': self.learning_rate, 'test_trajectory_starts': self.test_trajectory_starts}

        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        fileName = 'checkpoints\\' + fileName + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, fileName)

    def load_agent(self, fileName):
        """Load the checkpoint"""
        # print("\nLoading checkpoint\n")
        filepath = 'checkpoints\\' + fileName + '.pth'

        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.best_reward = checkpoint['best_reward']
            self.last_upgraded_frame = checkpoint['last_upgraded_frame']
            self.learning_rate = checkpoint['learning_rate']
            self.test_trajectory_starts = checkpoint['test_trajectory_starts']
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, eps = self.adam_eps)

            print("Loading checkpoint - Last Best Reward {} at Frame {} with LR {}".format(self.best_reward, self.last_upgraded_frame, self.learning_rate))
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(fileName))


class ReplayBuffer:
    """Fixed size buffer to store experience tuples"""
    
    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object
        
        Params
        ========
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["states", "prev_actions", "actions", "log_probs", "values", "returns", "advantage"])
        self.sub_states = []
        self.sub_next_states = []
        self.sub_prev_actions = []
        self.sub_actions = []
        self.sub_log_probs = []
        self.sub_values = []
        self.sub_rewards = []
        self.sub_masks = []
      
    def add(self, states, prev_actions, actions, log_probs, values, returns, advantage):
        """Add new experience to memory"""
        e = self.experience(np.asarray(self.sub_states), np.asarray(self.sub_prev_actions), np.asarray(self.sub_actions), np.asarray(self.sub_log_probs), np.asarray(self.sub_values), returns.cpu().numpy(), advantage.cpu().numpy())

        self.memory.append(e)
        self.clear_sub_buffer()

    def add_sub(self, state, next_state, prev_action, action, log_probs, value, reward, done):
        """Adds variables to sub buffer"""
        self.sub_states.append(state.cpu().numpy())
        self.sub_next_states.append(next_state.cpu().numpy())
        self.sub_prev_actions.append(prev_action.cpu().numpy())
        self.sub_actions.append(action)
        self.sub_log_probs.append(log_probs.detach().cpu().numpy())
        self.sub_values.append(value.detach().cpu().numpy())
        self.sub_rewards.append(reward)
        self.sub_masks.append(1 - done)
    
    def clear_sub_buffer(self):
        """Clears the sub buffer"""
        self.sub_states = []
        self.sub_next_states = []
        self.sub_prev_actions = []
        self.sub_actions = []
        self.sub_log_probs = []
        self.sub_values = []
        self.sub_rewards = []
        self.sub_masks = []

        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experience = random.sample(self.memory, k = 1)
        states, prev_actions, actions, log_probs, values, returns, advantage = zip(*experience)

        states = torch.squeeze(torch.from_numpy(experience[0].states).float().to(self.device))
        prev_actions = torch.squeeze(torch.from_numpy(experience[0].prev_actions).float().to(self.device))
        actions = torch.from_numpy(experience[0].actions).float().to(self.device)
        log_probs = torch.from_numpy(experience[0].log_probs).float().to(self.device)
        values = torch.from_numpy(experience[0].values).float().to(self.device)
        returns = torch.from_numpy(experience[0].returns).float().to(self.device)
        advantage = torch.from_numpy(experience[0].advantage).float().to(self.device)

        return states, prev_actions, actions, log_probs, values, returns, advantage


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
