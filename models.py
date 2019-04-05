import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.act_fn = params['act_fn']
        self.batchnorm = params['norm']
        
        hidden_layers = params['hidden_layers']
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], self.action_size)
        self.dropout = nn.Dropout(p = params['dropout'])

        self.norms = []
        if self.batchnorm:
            for linear in self.hidden_layers:
                self.norms.append(nn.LayerNorm(linear.in_features).to(device))
        
        initialize_weights(self, WEIGHT_LOW, WEIGHT_HIGH)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for i, linear in enumerate(self.hidden_layers):
            if self.batchnorm:
                x = self.norms[i](x)
            x = linear(x)
            x = self.act_fn[i](x)
            x = self.dropout(x)
        
        x = self.act_fn[-1](self.output(x))
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, params):
        """Initialize parameters and build model.
        Params
        ======
            params (dict-lie): dictionary of parameters
        """
        super(Critic, self).__init__()

        
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.seed = torch.manual_seed(params['seed'].next())
        self.act_fn = params['act_fn']
        self.action_layer = params['action_layer']
        self.batchnorm = params['norm']
        
        hidden_layers = params['hidden_layers'].copy()

        self.norms = [nn.LayerNorm(self.state_size).to(device)]
        if self.batchnorm:
            for hidden_neurons in hidden_layers:
                self.norms.append(nn.LayerNorm(hidden_neurons).to(device))
                
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        hidden_layers[self.action_layer-1] += self.action_size
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.dropout = nn.Dropout(p = params['dropout'])

        initialize_weights(self, WEIGHT_LOW, WEIGHT_HIGH)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = state
        for i, linear in enumerate(self.hidden_layers):
            if self.batchnorm:
                x = self.norms[i](x)
            if i == self.action_layer:
                x = torch.cat((x, action), dim=1)
            x = linear(x)
            x = self.act_fn[i](x)
            x = self.dropout(x)
        
        if self.batchnorm:
            x = self.norms[i+1](x)
        x = self.act_fn[-1](self.output(x))
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

        
        hidden_layers = params['hidden_layers'].copy()

        self.norms = [nn.LayerNorm(self.state_size).to(device)]
        if self.batchnorm:
            for hidden_neurons in hidden_layers:
                self.norms.append(nn.LayerNorm(hidden_neurons).to(device))
                
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        hidden_layers[self.action_layer-1] += self.num_atoms
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], 1)
        self.dropout = nn.Dropout(p = params['dropout'])

        initialize_weights(self, WEIGHT_LOW, WEIGHT_HIGH)


    def forward(self, state, action, log = False):
        for i, linear in enumerate(self.hidden_layers):
            if self.batchnorm:
                x = self.norms[i](state)
            if i == self.num_atoms:
                x = torch.cat((x, action), dim=1)
            x = linear(x)
            x = self.act_fn[i](x)
            x = self.dropout(x)
        
        if self.batchnorm:
            x = self.norms[i+1](x)
        x = self.act_fn[-1](self.output(x))

        # Only calculate the type of softmax needed by the foward call, to save
        # a modest amount of calculation across 1000s of timesteps.
        if log:
            return F.log_softmax(x, dim=-1)
        else:
            return F.softmax(x, dim=-1)
