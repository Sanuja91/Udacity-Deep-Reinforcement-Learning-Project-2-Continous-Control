""" Contains a variety of neural network architectures to use on the problem"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable

import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, mini_batch_size, num_channels = 3, std = 0.0):  # num_channels is 3, because its close close, high, low
        super(ActorCritic, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(state_size, 100)      
        self.fc2 = nn.Linear(100, 500)                     
        self.fc3 = nn.Linear(500, self.process_out)                     # add the weights from the previous timestep    

        self.critic = nn.Sequential(
            nn.Linear(self.process_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(self.process_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        self.apply(init_weights)
        print("############# LOG STD", self.log_std)
        

    def forward(self, state, prev_action, debug = False):
        x = F.relu(self.fc1(state))
        if debug != False:
            print("##### FC 1", x.shape)

        x = F.relu(self.fc2(x))
        if debug != False:
            print("##### FC 2", x.shape, "PREV ACTION", prev_action.shape)

        x = F.relu(self.fc3(x))
        if debug != False:
            print("##### FC 3", x.shape)

        value = self.critic(x)
        if debug != False:
            print("##### CRITIC VALUE", value.shape)

        mu = self.actor(x)
        mu.unsqueeze_(0)
        if debug != False:
            print("##### ACTOR MU", mu.shape)

        std = self.log_std.exp().expand_as(mu)
        if debug != False:
            print("## STD ", std.shape)

        dist = Normal(mu, std)
        sample = dist.sample()
        if debug != False:
            print("################# DIST ", sample)
            
        return dist, value, sample

    def get_action(self, sample):
        action = F.softmax(sample, dim=1)
        return np.ndarray.flatten(action.cpu().numpy())
