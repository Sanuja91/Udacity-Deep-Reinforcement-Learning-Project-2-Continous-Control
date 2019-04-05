import numpy as np

class GaussianNoise(object):
    def __init__(self, dimension, num_epochs, mu=0.0, var=1):
        self.mu = mu
        self.var = var
        self.dimension = dimension
        self.epochs = 0
        self.num_epochs = num_epochs
        self.min_epsilon = 0.01 # minimum exploration probability
        self.epsilon = 0.3
        self.decay_rate = 5.0/num_epochs # exponential decay rate for exploration prob
        self.iter = 0

    def sample(self):
        x = self.epsilon * np.random.normal(self.mu, self.var, size=self.dimension)
        return x

    def reset(self):
        self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*self.iter)
        
class OrnsteinUhlenbeckProcess(object):
    def __init__(self, dimension, num_steps, theta=0.25, mu=0.0, sigma=0.05, dt=0.01):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x = np.zeros((dimension,))
        self.iter = 0
        self.num_steps = num_steps
        self.dimension = dimension
        self.min_epsilon = 0.01 # minimum exploration probability
        self.epsilon = 1.0
        self.decay_rate = 5.0/num_steps # exponential decay rate for exploration prob
    
    def sample(self):
        self.x = self.x + self.theta*(self.mu-self.x)*self.dt + \
                                       self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.dimension)
        return self.epsilon*self.x
    
    def reset(self):
        self.x = 0*self.x
        self.iter += 1
        self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon)*np.exp(-self.decay_rate*self.iter)
