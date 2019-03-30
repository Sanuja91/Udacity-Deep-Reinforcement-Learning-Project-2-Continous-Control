import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical

import numpy as np

# Initializing the weights of the neural network in an optimal way for the learning
def init_weights(m):
    classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if classname.find('Conv') != -1: # if the connection is a convolution
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif classname.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, std = 0.0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        FC1 = 50
        FC2 = 100
        FC3 = 100
        FC4 = 50

        
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, FC1),
            nn.LayerNorm(FC1),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(FC1, FC2),
            nn.LayerNorm(FC2),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(FC2, FC3),
            nn.LayerNorm(FC3),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(FC3, FC4),
            nn.LayerNorm(FC4),
            nn.ReLU()
        )


        self.fc5 = nn.Sequential(
            nn.Linear(FC4, action_size),
            nn.LayerNorm(action_size),
            nn.Tanh()
        )


        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, state, debug = False):
        """Build an actor (policy) network that maps states -> actions."""
        # state = state.unsqueeze(0)
        if debug:
            print("STATE", state)
        x = self.fc1(state)
        if debug:
            print("FC1", x.shape)
        x = self.fc2(x)
        if debug:
            print("FC2", x.shape)
        x = self.fc3(x)
        if debug:
            print("CAT", x.shape)
        x = self.fc4(x)
        if debug:
            print("FC4", x.shape)
        mu = self.fc5(x).unsqueeze_(0)
        if debug:
            print("FC5 - MU", mu.shape)

        std = self.log_std.exp().expand_as(mu)

        distribution = Normal(mu, std)
        if debug:
            print("DISTRIBUTION", distribution)

        # if debug:
        #     print("ACTION", action)
        return distribution

    def get_actions(self, distribution, n):
        actions = distribution.sample().float().squeeze(0)
        return actions

        # print(action)    

        # exit()
        # return action


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        FCS1 = 50
        FCS2 = 100
        FCS3 = 4
        FC4 = 24

        self.fcs1 = nn.Sequential(
            nn.Linear(state_size, FCS1),
            nn.LayerNorm(FCS1),
            nn.ReLU()
        )

        self.fcs2 = nn.Sequential(
            nn.Linear(FCS1, FCS2),
            nn.LayerNorm(FCS2),
            nn.ReLU()
        )

        self.fcs3 = nn.Sequential(
            nn.Linear(FCS2, FCS3),
            nn.LayerNorm(FCS3),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(FCS3 + action_size, FC4),
            nn.LayerNorm(FC4),
            nn.ReLU()
        )


        self.fc5 = nn.Sequential(
            nn.Linear(FC4, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_weights)

    def forward(self, states, actions, debug = False):
        """Build a critic (value) network that maps (states, actions) pairs -> Q-values."""
        if debug:
            print("STATES", states.shape)
        xs = self.fcs1(states)
        if debug:
            print("FSC1", xs.shape)
        xs = self.fcs2(xs)
        if debug:
            print("FSC2", xs.shape)
        xs = self.fcs3(xs)
        # actions.squeeze_(0)
        if debug:
            print("FSC3", xs.shape, "ACTION", actions)
        x = torch.cat((xs, actions), dim=1)
        if debug:
            print("CAT", x.shape)
        x = self.fc4(x)
        if debug:
            print("FC4", x.shape)
        x = self.fc5(x)
        if debug:
            print("FC5", x.shape)
        return x