import numpy as np
import random
import copy
import os


from models import Actor, Critic

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
    def __init__(self, name, id, device, state_size, action_size, load_agent = False):        

        self.device = device

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(RANDOM_SEED)
        self.name = name
        self.id = id

        # Hyperparameters
        self.gamma = GAMMA
        self.tau = TAU
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.weight_decay = LEARNING_RATE_DECAY

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, RANDOM_SEED).to(self.device)
        self.actor_target = Actor(state_size, action_size, RANDOM_SEED).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, RANDOM_SEED).to(self.device)
        self.critic_target = Critic(state_size, action_size, RANDOM_SEED).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        if load_agent:
            self.load_agent(self.name)

        # Noise process
        self.noise = OUNoise(action_size, RANDOM_SEED)
    
    def step(self, state, action, reward, next_state, done, shared_memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        # Save experience / reward
        shared_memory.add(state, action, reward, next_state, done)

    def act(self, state, add_noise = True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
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
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
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

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        tau = self.tau
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_agent(self):
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
            self.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_local.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
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
