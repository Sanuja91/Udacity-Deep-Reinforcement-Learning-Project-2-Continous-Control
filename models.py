import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, device, std = 0.0, dropout_rate = 0.5):
        super(ActorCritic, self).__init__()
        
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

        FC1 = 100
        FC2 = 150
        FC3 = 200

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_size, FC1),
            nn.LayerNorm(FC1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(FC1, FC2),
            nn.LayerNorm(FC2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(FC2, FC3),                             # add the weights from the previous timestep    
            nn.LayerNorm(FC3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(FC3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(FC3, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, self.action_size),
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        self.apply(init_weights)
    
    def forward(self, states, debug = False):
        if debug:
            print("##### STATES", states.shape)

        x = self.fc1(states)
        if debug:
            print("##### FC 1", x.shape)

        x = self.fc2(x)
        # x = torch.cat((x.squeeze(0), prev_action), dim = 0)        # concats prev_actions to x
        # if debug:
        #     print("##### FC 2", x.shape, "PREV ACTION", prev_action.shape)

        x = self.fc3(x)
        if debug:
            print("##### FC 3", x.shape)

        values = self.critic(x)
        if debug:
            print("##### CRITIC VALUE", values.shape)

        mu = self.actor(x)
        # mu.unsqueeze_(0)
        if debug:
            print("##### ACTOR MU", mu.shape)

        return mu, values
