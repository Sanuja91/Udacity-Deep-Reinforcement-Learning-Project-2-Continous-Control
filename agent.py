# coding: utf-8
from abc import ABCMeta, abstractmethod

import random
import copy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from . model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
