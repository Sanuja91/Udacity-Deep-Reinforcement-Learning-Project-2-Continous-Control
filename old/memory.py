from collections import namedtuple, deque
import numpy as np
import torch
import random

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.priority = False
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def batch_passed(self):
        return len(self.memory) > self.batch_size


class TrajectoryReplayBuffer:
    """Fixed-size buffer to store n step trajectories."""

    def __init__(self, device, buffer_size, batch_size, seed, max_trajectory_length, min_trajectory_length, agent_ids):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.device = device
        self.priority = False
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "value", "log_prob"])
        self.seed = random.seed(seed)
        self.max_trajectory_length = max_trajectory_length
        self.min_trajectory_length = min_trajectory_length
        self.current_trajectory_length = 0
        self.agent_ids = agent_ids
        self.agent_trajectories = [[] for agent_id in self.agent_ids]    # used to build trajectories for each agent seperately during exploration
        
        # Randomly picks next tracjetory length to store
        self.current_trajectory_cut_off = random.randint(self.min_trajectory_length, self.max_trajectory_length)
    
    def buffer_experiences(self, states, actions, rewards, next_states, dones, values, log_probs):
        """Buffers experience for serveral agents forming trajectories and stores them when required
           Returns True if trajectory has reached terminal state
        """
        # print("states", states.shape, "actions", actions.shape, "rewards", len(rewards), "next_states", next_states.shape, "dones", len(dones), "values", values.shape, "log_probs", log_probs.shape)
        if self.current_trajectory_length == self.current_trajectory_cut_off or dones[0] == 1:  # checks current state is at a terminal state
            for agent_id in self.agent_ids:
                e = self.experience(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], 1, values[agent_id], log_probs[agent_id])   # forces terminal state at cut off
                self.agent_trajectories[agent_id].append(e)
            self.current_trajectory_length = self.current_trajectory_cut_off + 1 
            return False
        elif self.current_trajectory_length < self.current_trajectory_cut_off: # force break and clean buffer
            for agent_id in self.agent_ids:
                e = self.experience(states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], dones[agent_id], values[agent_id], log_probs[agent_id])
                self.agent_trajectories[agent_id].append(e)
            self.current_trajectory_length += 1
            return False
        else:
            self._store_trajectories()
            return True


    def _store_trajectories(self):
        """Add new trajectories to memory and resets counter and makes new n step cut off."""   
        for trajectory in self.agent_trajectories:
            self.memory.append(trajectory)

        self.current_trajectory_length = 0
        self.current_trajectory_cut_off = random.randint(self.min_trajectory_length, self.max_trajectory_length)
        self.agent_trajectories = [[] for agent_id in self.agent_ids]

    
    def sample_trajectories(self):
        """Randomly sample several stored trajectories and yield experiences of each trajectory"""
        trajectories = random.sample(self.memory, k = self.batch_size)
        for experiences in trajectories:
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
            values = torch.from_numpy(np.vstack([e.value for e in experiences if e is not None])).float().to(self.device)
            log_probs = torch.from_numpy(np.vstack([e.log_prob for e in experiences if e is not None])).float().to(self.device)
            yield (states, actions, rewards, next_states, dones, values, log_probs)

    def batch_passed(self):
        return len(self.memory) > self.batch_size

class NaivePrioritizedBuffer(object):
    def __init__(self, device, buffer_size, batch_size, prob_alpha = 0.6):
        self.device = device
        self.prob_alpha = prob_alpha
        self.priority = True
        self.buffer_size   = buffer_size
        self.memory     = []
        self.batch_size = batch_size
        self.pos        = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        # assert state.ndim == next_state.ndim
        # state      = np.expand_dims(state, 0)
        # next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.priorities.max() if self.memory else 1.0
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size
    
    def sample(self, beta=0.4):
        if len(self.memory) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        
        total    = len(self.memory)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        # batch       = zip(*samples)
        # states      = np.concatenate(batch[0])
        # actions     = batch[1]
        # rewards     = batch[2]
        # next_states = np.concatenate(batch[3])
        # dones       = batch[4]
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def batch_passed(self):
        return len(self.memory) > self.batch_size