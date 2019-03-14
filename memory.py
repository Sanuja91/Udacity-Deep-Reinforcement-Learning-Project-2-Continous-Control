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

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

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

    def __len__(self):
        return len(self.memory)

import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        print("####\n\n", self.tree)
        return self.tree[0]

    def next_idx(self):
        return self.write + self.capacity - 1

    def add(self, p, data):
        idx = self.next_idx()

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedBuffer:   # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, device, buffer_size, batch_size):
        self.device = device
        self.priority = True
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["idx", "state", "action", "reward", "next_state", "done"])
        self.tree = SumTree(self.buffer_size)
        self.error_max = 0
        self.e = 0.01
        self.a = 0.6

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, state, action, reward, next_state, done, error):
        idx = self.tree.next_idx()
        e = self.experience(idx, state, action, reward, next_state, done)
        # print("######## ADD EXPERIENCE\n", e)
        p = self._get_priority(error)
        # print("######## PRIORITY\n", p)
        self.tree.add(p, e) 

    def batch_passed(self):
        segment = self.tree.total() / self.batch_size
        return segment > 1

    def sample(self):
        experiences = []
        segment = self.tree.total() / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, experience) = self.tree.get(s)
            experiences.append(experience)
            # experiences.append( (idx, experience) )

        print("################ EXPERIENCES\n", experiences)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        indices = np.vstack([e.done for e in experiences if e is not None])
        return (states, actions, rewards, next_states, dones, indices)

    def update(self, idx, error):
        p = self._get_priority(error)
        if error > self.error_max:
            self.error_max = error
        self.tree.update(idx, p)
