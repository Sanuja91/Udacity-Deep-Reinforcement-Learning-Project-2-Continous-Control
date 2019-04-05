import torch, math
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

        FC1 = 128
        FC2 = 128
        FC3 = 50

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_size, FC1),
            nn.LayerNorm(FC1),
            nn.ELU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(FC1, FC2),
            nn.LayerNorm(FC2),
            nn.ELU(),
            # # nn.Dropout(dropout_rate)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(FC2, FC3),                             # add the weights from the previous timestep    
            nn.LayerNorm(FC3),
            nn.ELU(),
            # # nn.Dropout(dropout_rate)
        )
        print("############## \n\nCRITIC DOESNT FACTOR IN ACTION!!!")
        exit()
        
        self.critic = nn.Sequential(
            nn.Linear(FC3, FC3),
            nn.LayerNorm(FC3),
            nn.ELU(),
            nn.Linear(FC3, FC3),
            nn.LayerNorm(FC3),
            nn.ELU(),
            nn.Linear(FC3, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(FC3, FC3),
            nn.LayerNorm(FC3),
            nn.ELU(),
            nn.Linear(FC3, FC3),
            nn.LayerNorm(FC3),
            nn.ELU(),
            # NoisyFactorizedLinear(FC3, self.action_size),
            nn.Linear(FC3, self.action_size),
            nn.LayerNorm(self.action_size),
            nn.ELU(),
        )

        self.log_std = nn.Parameter(torch.ones(1, action_size) * std)
        # self.apply(init_weights)
    
    def forward(self, states, debug = False):
        if debug:
            print("##### STATES", states.shape)

        x = self.fc1(states)
        if debug:
            print("##### FC 1", x.shape)

        x = self.fc2(x)
        if debug:
            print("##### FC 2", x.shape)

        x = self.fc3(x)
        if debug:
            print("##### FC 3", x.shape)

        values = self.critic(x)
        if debug:
            print("##### CRITIC VALUE", values.shape)

        mu = self.actor(x)

        if debug:
            print("##### ACTOR MU", mu.shape)

        return mu, values

# class NoisyFactorizedLinear(nn.Linear):
#     """
#     NoisyNet layer with factorized gaussian noise
#     N.B. nn.Linear already initializes weight and bias to
#     """
#     def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
#         super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
#         sigma_init = sigma_zero / math.sqrt(in_features)
#         self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
#         self.register_buffer("epsilon_input", torch.zeros(1, in_features))
#         self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
#         if bias:
#             self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

#     def forward(self, input):
#         torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
#         torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

#         func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
#         eps_in = func(self.epsilon_input)
#         eps_out = func(self.epsilon_output)

#         bias = self.bias
#         if bias is not None:
#             bias = bias + self.sigma_bias * Variable(eps_out.t())
#         noise_v = Variable(torch.mul(eps_in, eps_out))

#         return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)
