import numpy as np
import random
import torch
from collections import namedtuple, deque

from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device

BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128       
RANDOM_SEED = 4

def ensure_shared_grads(model, shared_model):
    """"""
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def share_learning(shared_model, agents):
    """Distribute the learning across all agents"""
    new_agents = []
    for agent in agents:
        ensure_shared_grads(agent.actor_local, shared_model)
        new_agents.append(agent)

    return new_agents


def ddpg(multiple_agents = False, PER = False, n_episodes = 300, max_t = 1000):
    """ Deep Deterministic Policy Gradients
    Params
    ======
        multiple_agents (boolean): boolean for multiple agents
        PER (boolean): 
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    env, env_info, states, state_size, action_size, brain_name, num_agents = initialize_env(multiple_agents)
    
    device = get_device()
    scores_window = deque(maxlen=100)
    scores = np.zeros(num_agents)
    scores_episode = []
    
    agents = [] 
    shared_memory = NaivePrioritizedBuffer(device, BUFFER_SIZE, BATCH_SIZE, RANDOM_SEED)
    for agent_id in range(num_agents):
        agents.append(Actor_Crtic_Agent(brain_name, agent_id, device, state_size, action_size))
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = True)[brain_name]
        states = env_info.vector_observations
        
        for agent in agents:
            agent.reset()
            
        scores = np.zeros(num_agents)
            
        for t in range(max_t):      
            actions = np.array([agents[i].act(states[i]) for i in range(num_agents)])
            env_info = env.step(actions)[brain_name]       # send the action to the environment
            next_states = env_info.vector_observations     # get the next state
            rewards = env_info.rewards                     # get the reward
            dones = env_info.local_done        
            
            for i in range(num_agents):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], shared_memory) 
            if len(shared_memory.memory) > BATCH_SIZE:
                experiences = shared_memory.sample()
                agents[0].learn(experiences, shared_memory)
                agents = share_learning(agents[0].actor_local, agents)
 
            states = next_states
            scores += rewards
            if t % 20:
                print('\rTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(t, np.mean(scores), np.min(scores), np.max(scores)), end="") 
            if np.any(dones):
                break 
        score = np.mean(scores)
        scores_window.append(score)       # save most recent score
        scores_episode.append(score)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores_window)), end="\n")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            # torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores_episode



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
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
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
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)