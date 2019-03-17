import numpy as np
import matplotlib.pyplot as plt

from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device
from ddpg import ddpg

AGENT_NAME = "FC ONLY - BASIC REPLAY"
MULTI = False   # Multiple agents or single

if MULTI:
    AGENT_NAME = "MULTI AGENT - " + AGENT_NAME
else:
    AGENT_NAME = "SINGLE AGENT - " + AGENT_NAME

scores = ddpg(AGENT_NAME, multiple_agents = MULTI)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()