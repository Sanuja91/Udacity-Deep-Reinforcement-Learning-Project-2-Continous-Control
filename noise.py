import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, params):
        """Initialize parameters and noise process."""

        mu = params['mu']
        theta = params['theta']
        sigma = params['sigma']
        seed = params['seed']
        size = params['action_size']
        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed.next())
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def create_noise(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class GaussianExploration:
    def __init__(self, params):
        self.epsilon = params['max_epsilon']
        self.min_epsilon = params['min_epsilon']
        self.decay_rate = params['decay_rate']
    
    def create_noise(self, shape):
        self.epsilon  = max(self.epsilon, self.min_epsilon)  
        self.epsilon *= self.decay_rate     # decay epsilon
        return np.random.normal(0, 1, shape) * self.epsilon