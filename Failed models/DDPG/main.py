import numpy as np
import matplotlib.pyplot as plt

from utilities import initialize_env, get_device
from ddpg import batch_ddpg

AGENT_NAME = "FC SEQ MED - BASIC REPLAY - BATCH DDPG"
MULTI = True   # Multiple agents or single

if MULTI:
    AGENT_NAME = "MULTI AGENT - " + AGENT_NAME
else:
    AGENT_NAME = "SINGLE AGENT - " + AGENT_NAME

scores = batch_ddpg(AGENT_NAME, multiple_agents = MULTI)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()