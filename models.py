import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utilities import get_device

device = get_device()

WEIGHT_LOW = -3e-2
WEIGHT_HIGH = 3e-2

def initialize_weights(model, low, high):
    for param in model.parameters():
        param.data.uniform_(low, high)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
            params (dict-lie): dictionary of parameters
        """
        super(Actor, self).__init__()
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.seed = torch.manual_seed(params['seed'].next())

        FC1 = 400
        FC2 = 300
        FC3 = 128
        FC4 = 64

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_size, FC1),
            # nn.LayerNorm(FC1),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(FC1, FC2),
            # nn.LayerNorm(FC2),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(FC2, FC3),                               
            # nn.LayerNorm(self.action_size),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(FC3, FC4),                               
            # nn.LayerNorm(self.action_size),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(FC4, self.action_size),                             # add the weights from the previous timestep    
            # nn.LayerNorm(self.action_size),
            nn.Tanh(),
            # # nn.Dropout(dropout_rate)
        )

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


# Set up critic network in D4PG (excerpted).
class D4PGCritic(nn.Module):
    """
    The critic network  approximates the Value (V) of the suggested actions produced 
    by the Actor network.
    """

    def __init__(self, params):
        super(D4PGCritic, self).__init__()

        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.seed = torch.manual_seed(params['seed'].next())
        self.act_fn = params['act_fn']
        self.action_layer = params['action_layer']
        self.batchnorm = params['norm']
        self.num_atoms = params['num_atoms']
        
        FC1 = 400
        FC2 = 300
        FC3 = 128
        FC4 = 64

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_size, FC1),
            # nn.LayerNorm(FC1),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(FC1, FC2),
            # nn.LayerNorm(FC2),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(FC2, FC3),
            # nn.LayerNorm(FC2),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(FC3, FC4),
            # nn.LayerNorm(FC2),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc5 = nn.Sequential(
            nn.Linear(FC4 + self.action_size, self.num_atoms),                             # add the weights from the previous timestep    
            # nn.LayerNorm(self.num_atoms),
            nn.ReLU(),
            # # nn.Dropout(dropout_rate)
        )

        initialize_weights(self, WEIGHT_LOW, WEIGHT_HIGH)

    def forward(self, state, action, log = False):

        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc5(x)     # here x is equivilent to logits

        
        # Only calculate the type of softmax needed by the foward call, to save
        # a modest amount of calculation across 1000s of timesteps.
        if log:
            return F.log_softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)

        return x
