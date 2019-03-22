import numpy as np
import random
import copy
import os


from models2 import Actor, Critic
from memory import TrajectoryReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
TAU = 1e-2
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
LEARNING_RATE_DECAY = 5e-8
RANDOM_SEED = 4 

class Actor_Crtic_Agent():        
    def __init__(self, name, id, device, state_size, action_size, num_agents, buffer_size, batch_size, seed, max_trajectory_size, min_trajectory_size, load_agent = False):        

        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.num_agents = num_agents
        self.name = name
        self.id = id
        self.best_reward = 0
        self.memory = TrajectoryReplayBuffer(self.device, buffer_size, batch_size, seed, max_trajectory_size, min_trajectory_size, range(self.num_agents))
        
        # Hyperparameters
        self.gamma = GAMMA
        self.tau = TAU
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.weight_decay = LEARNING_RATE_DECAY

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, RANDOM_SEED).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = Critic(state_size, action_size, RANDOM_SEED).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        if load_agent:
            self.load_agent(self.name)

        # Noise process
        self.noise = OUNoise(action_size, RANDOM_SEED)
    
    def step(self, states, actions, rewards, next_states, dones, values, log_probs):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        terminal_state_reached = self.memory.buffer_experiences(states, actions, rewards, next_states, dones, values, log_probs) 
        
        if np.max(rewards) > self.best_reward:
            self.best_reward = np.max(rewards) 

        # Buffers experiences and checks if terminal state has been reached and if enough batches are there to learn
        if terminal_state_reached and self.memory.batch_passed:
            self._learn()


    def act(self, states, add_noise = True, train = True):
        """Returns actions for given states as per current policy."""
        states = torch.from_numpy(states).float().to(self.device)
        if train:
            self.actor.train()        
            distribution = self.actor(states)
        else:
            self.actor.eval()
            with torch.no_grad():
                distribution = self.actor(states)
            self.actor.train()

        actions = self.actor.get_actions(distribution, n = self.num_agents)
        log_probs = distribution.log_prob(actions.squeeze(0)).squeeze(0).cpu().data.numpy()
        
        print("ACTIONS", actions.cpu().data.numpy().shape, "LOG PROBS", log_probs.shape)
        values = self.critic(states, actions).cpu().data.numpy()
    
        actions = actions.cpu().data.numpy()
        if add_noise:
            noise = self.noise.sample()
            actions += noise
        return np.clip(actions, -1, 1), values, log_probs

    def reset(self):
        self.noise.reset()

    def _learn(self):
        """Samples trajectories and learns from them"""
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
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
            self.best_reward = checkpoint['best_reward']

            print("Loading checkpoint - Last Best Reward {} (%) at Frame {} with LR {}".format((np.exp(self.best_reward) - 1) * 100, self.last_upgraded_frame, self.learning_rate))
        else:
            print("\nCannot find {} checkpoint... Proceeding to create fresh neural network\n".format(fileName))        

    def _estimate_advantages(self, rewards, masks, values, gamma, tau, device):
        """Calculates generalized advantage estimates] (GAE)"""
        
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)

        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages
        advantages = (advantages - advantages.mean()) / advantages.std()

        advantages, returns = advantages.to(self.device), returns.to(self.device)
        return advantages, returns

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
