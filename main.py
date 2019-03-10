import numpy as np
import matplotlib.pyplot as plt

from agents import Actor_Crtic_Agent
from interact import interact
from utilities import initialize_env, get_device
from ddpg import ddpg

NUM_AGENTS = 1

scores = ddpg(NUM_AGENTS)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()