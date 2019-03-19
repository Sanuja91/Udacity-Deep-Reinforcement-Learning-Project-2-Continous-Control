import numpy as np
import matplotlib.pyplot as plt

from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device
from ddpg import ddpg, batch_ddpg

MULTI = True   # Multiple agents or single

scores = batch_ddpg(MULTI)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()