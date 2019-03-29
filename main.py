import numpy as np
import matplotlib.pyplot as plt

from utilities import initialize_env, get_device
from a2c import actor_critic

### ADD LEARNING RATE DECAY

AGENT_NAME = "PROPER A2C"
MULTI = True   # Multiple agents or single

if MULTI:
    AGENT_NAME = "MULTI AGENT - " + AGENT_NAME
else:
    AGENT_NAME = "SINGLE AGENT - " + AGENT_NAME


scores = actor_critic(AGENT_NAME, multiple_agents = MULTI)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()