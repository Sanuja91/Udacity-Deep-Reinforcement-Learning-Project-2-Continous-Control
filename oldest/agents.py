import numpy as np
import random
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable
from torch.distributions import Categorical

GAMMA = 0.99
GAE_LAMBDA = 0.95
TAU = 1e-2
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
LEARNING_RATE_DECAY = 5e-8
RANDOM_SEED = 4 

class AC_Agent():        
    def __init__(self, name, id, device, state_size, action_size, num_agents, seed, max_trajectory_size, min_trajectory_size, load_agent = False):        

        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.num_agents = num_agents
        self.name = name
        self.id = id
        self.best_reward = 0
        
        # Hyperparameters
        self.gamma = GAMMA
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.weight_decay = LEARNING_RATE_DECAY

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, RANDOM_SEED).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = self.lr_actor, weight_decay = self.weight_decay)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size, action_size, RANDOM_SEED).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = self.lr_critic, weight_decay = self.weight_decay)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.lob_probs = []
        self.returns = []
        self.advantages = []

        if load_agent:
            self.load_agent(self.name)

        # # Noise process
        # self.noise = OUNoise(action_size, RANDOM_SEED)
    
    def step(self, state, action, reward, next_state, done, value, log_prob):
        """Agent steps through the environment step and learns from experience"""

        if done == 1:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.values.append(value)
            self.lob_probs.append(lob_prob)
            
            next_value = self.critic(next_state)
            masks = 1 - self.dones
            self.returns, self.advantages = self._compute_gae(next_value, self.rewards, masks, self.values)
            
            self._learn()

            self.states = []
            self.actions = []
            self.rewards = []
            self.next_states = []
            self.dones = []
            self.values = []
            self.lob_probs = []

        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.values.append(value)
            self.lob_probs.append(lob_prob)


    def act(self, state, add_noise = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        self.actor.train()        
        distribution = self.actor(state)

        action = self.actor.get_action(distribution)
        log_prob = distribution.log_prob(action.squeeze(0)).squeeze(0).cpu().data.numpy()
        
        print("ACTION", action.cpu().data.numpy().shape, "LOG PROB", log_prob.shape)
        value = self.critic(state).cpu().data.numpy()
    
        action = action.cpu().data.numpy()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1), value, log_prob

    # def reset(self):
    #     self.noise.reset()


    def _compute_gae(self, next_value, rewards, masks, values, gamma = GAMMA, lamda_ = GAE_LAMBDA):
        """Compute Generalized Advantage Estimation"""
        # print("####################################### COMPUTING GAE")
        returns = []
        gae_returns = []
        gae_advantages = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            returns = rewards[step] + lamda_ * masks[step] * returns
            td_error = rewards[step] + lamda_ * masks[step] * (values[step + 1] - values[step])
            advantages = advantages * gamma * lamda_ * masks[step] + td_error
            gae_returns[step] = returns
            gae_advantages[step] = advantages
            # returns.insert(0, return_)
        return gae_returns, gae_advantages

    def _learn(self):
        """Learn from latest trajectory"""
        distribution = self.actor()
        new_action = self.actor.get_action(distribution)

        new_values = self.critic(self.states)




        for states, actions, rewards, next_states, dones, values, log_probs in self.memory.sample_trajectories():
            masks = 1 - dones
            advantages, returns = self._estimate_advantages(rewards, masks, values, GAMMA, TAU, self.device)

            # (policy_net, value_net, optimizer_policy, optimizer_value, states, actions, returns, advantages, l2_reg):
            
            distribution = self.actor(states)
            new_actions = self.actor.get_actions(distribution, n = self.num_agents)
            # log_probs = distribution.log_prob(actions).squeeze(0)
            # values = self.critic(states, actions).cpu().data.numpy()

            """Update critic"""
            values_pred = self.critic(states, new_actions)
            value_loss = (values_pred - returns).pow(2).mean()

            # weight decay
            # for param in value_net.parameters():
            #     value_loss += param.pow(2).sum() * l2_reg
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            """Update policy"""
            # log_probs = policy_net.get_log_prob(states, actions)
            policy_loss = -(log_probs * advantages).mean()
            # policy_loss = -values_pred.mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
            self.actor_optimizer.step()     


    def save_agent(self, fileName):
        """Save the checkpoint"""
        checkpoint = {'actor_state_dict': self.actor_target.state_dict(),'critic_state_dict': self.critic_target.state_dict(), 'best_reward': self.best_reward}
        
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints") 
        
        filePath = 'checkpoints\\' + fileName + '.pth'
        # print("\nSaving checkpoint\n")
        torch.save(checkpoint, filePath)

    def load_agent(self, fileName):
        """Load the checkpoint"""
        # print("\nLoading checkpoint\n")
        filePath = 'checkpoints\\' + fileName + '.pth'

        if os.path.exists(filePath):
            checkpoint = torch.load(filePath, map_location = lambda storage, loc: storage)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.best_reward = checkpoint['best_reward']

            print("Loading checkpoint - Last Best Reward {} (%) at Frame {} with LR {}".format((np.exp(self.best_reward) - 1) * 100, self.last_upgraded_frame, self.learning_rate))
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(fileName))        

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
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

        return distribution

    def get_action(self, distribution):
        action = distribution.sample().float().squeeze(0)
        return action

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

#### ELIGIBILITY TRACE ########################3

        # import numpy as np

# state_values = np.zeros(n_states) # initial guess = 0 value
# eligibility = np.zeros(n_states)

# lamb = 0.95 # the lambda weighting factor
# state = env.reset() # start the environment, get the initial state
# # Run the algorithm for some episodes
# for t in range(n_steps):
#   # act according to policy
#   action = policy(state)
#   new_state, reward, done = env.step(action)
#   # Update eligibilities
#   eligibility *= lamb * gamma
#   eligibility[state] += 1.0

#   # get the td-error and update every state's value estimate
#   # according to their eligibilities.
#   td_error = reward + gamma * state_values[new_state] - state_values[state]
#   state_values = state_values + alpha * td_error * eligibility

#   if done:
#     state = env.reset()
#   else:
#     state = new_state