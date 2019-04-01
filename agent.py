# coding: utf-8
from abc import ABCMeta, abstractmethod

import random
import copy
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(object):
    """Interacts with and learns from the environment."""
    __metaclass__ = ABCMeta
    
    def __init__(self, params):
        """Initialize an Agent object given a dictionary of parameters.
        
        Params
        ======
        * **params** (dict-like) --- a dictionary of parameters
        """

        self.params = params
        self.tau = params['tau']
        
        # Replay memory: to be defined in derived classes
        self.memory = None
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @abstractmethod
    def act(self, state, action):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        * **state** (array_like) --- current state
        * **action** (array_like) --- the action values
        """
        pass


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
        * **local_model** (PyTorch model) --- weights will be copied from
        * **target_model** (PyTorch model) --- weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    @abstractmethod
    def step(self, states, actions, rewards, next_states, dones):
        """Perform a step in the environment given a state, action, reward,
        next state, and done experience.
        Params
        ======
        * **states** (torch.Variable) --- the current state
        * **actions** (torch.Variable) --- the current action
        * **rewards** (torch.Variable) --- the current reward
        * **next_states** (torch.Variable) --- the next state
        * **dones** (torch.Variable) --- the done indicator
        * **betas** (float) --- a potentially tempered beta value for prioritzed replay sampling
        """
        pass

    @abstractmethod
    def learn_(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
        * **experiences** (Tuple[torch.Variable]) --- tuple of (s, a, r, s', done) tuples 
        """
        pass


class DDPGAgent(Agent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, idx, params):
        """Initialize an Agent object.
        
        Params
        ======
            params (dict-like): dictionary of parameters for the agent
        """
        super().__init__(params)

        self.idx = idx
        self.params = params
        self.update_every = params['update_every']
        self.gamma = params['gamma']
        self.num_agents = params['num_agents']
        self.name = "BATCH D4PG"
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(params['actor_params']).to(device)
        self.actor_target = Actor(params['actor_params']).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['actor_params']['lr'])
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(params['critic_params']).to(device)
        self.critic_target = Critic(params['critic_params']).to(device)

        print("\n################ ACTOR ################\n")
        print(self.actor_local)
        
        print("\n################ CRITIC ################\n")
        print(self.critic_local)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=params['critic_params']['lr'],
                                           weight_decay=params['critic_params']['weight_decay'])

        # Noise process
        self.noise = OUNoise(self.params['noise_params'])

        # Replay memory
        self.memory = params['experience_replay']
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # next_state = torch.from_numpy(next_states[self.idx]).float().unsqueeze(0).to(device)
        # state = torch.from_numpy(states[self.idx]).float().unsqueeze(0).to(device)
        
        # # print("\nSTATE\n", state, "\nACTION\n", actions[self.idx], "\nREWARD\n", rewards[self.idx], "\nNEXT STATE\n", next_state, "\nDONE\n", dones[self.idx])
        # # Save experience / reward
        # self.memory.add(state.cpu(), actions[self.idx], rewards[self.idx], next_state.cpu(), dones[self.idx])

        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # print("\nSTATE\n", state, "\nACTION\n", action, "\nREWARD\n", reward, "\nNEXT STATE\n", next_state, "\nDONE\n", done)
        # Save experience / reward
        self.memory.add(state.cpu(), action, reward, next_state.cpu(), done)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1., 1.)

    def reset(self):
        self.noise.reset()

    def learn(self):        
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        # print(self.t_step)
        # # self.t_step = (self.t_step + 1) % self.update_every
        # if self.t_step % self.update_every == 0:
        #     print("LEARNING", self.t_step)
        #     # If enough samples are available in memory, get random subset and learn
        #     if self.memory.ready():
        #         experiences = self.memory.sample()
        #         # print("################################## LEARN XP LENGTH",len(experiences))
        #         self.learn_(experiences)



        # If enough samples are available in memory, get random subset and learn
        if self.memory.ready():
            experiences = self.memory.sample()
            # print("################################## LEARN XP LENGTH",len(experiences))
            self.learn_(experiences)
        
        
    def learn_(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target) 


    def add_param_noise(self, noise):
        """Adds noise to the weights of the agent"""
        with torch.no_grad():
            for param in self.actor_local.parameters():
                param.add_(torch.randn(param.size()).to(device) * noise)
            for param in self.critic_local.parameters():
                param.add_(torch.randn(param.size()).to(device) * noise)


    def save_agent(self, average_reward, episode, save_history = False):
        """Save the checkpoint"""
        checkpoint = {'actor_state_dict': self.actor_target.state_dict(), 'critic_state_dict': self.critic_target.state_dict(), 'average_reward': average_reward, 'episode': episode}
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        filePath = 'checkpoints\\' + self.name + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, filePath)

        if save_history:
            filePath = 'checkpoints\\' + self.name + '_' + str(episode) + '.pth'
            torch.save(checkpoint, filePath)


    def load_agent(self):
        """Load the checkpoint"""
        # print("\nLoading checkpoint\n")
        filePath = 'checkpoints\\' + self.name + '.pth'

        if os.path.exists(filePath):
            checkpoint = torch.load(filePath, map_location = lambda storage, loc: storage)

            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])

            average_reward = checkpoint['average_reward']
            episode = checkpoint['episode']
            
            print("Loading checkpoint - Average Reward {} at Episode {}".format(average_reward, episode))
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(self.name))        
                    

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, params):
        """Initialize parameters and noise process."""

        mu = params['mu']
        theta = params['theta']
        sigma = params['sigma']
        seed = params['seed']
        size = params['action_size']
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed.next())
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state