import numpy as np
import matplotlib.pyplot as plt

from agents import Actor_Crtic_Agent
from utilities import initialize_env, get_device
<<<<<<< HEAD
from ddpg import ddpg, batch_ddpg
=======
from ddpg import ddpg
>>>>>>> f3a172e9d7954f8c7474f4d363a2cf7f4b85d97a

AGENT_NAME = "FC SEQ MED - BASIC REPLAY"
MULTI = True   # Multiple agents or single

<<<<<<< HEAD
scores = batch_ddpg(MULTI)
=======
if MULTI:
    AGENT_NAME = "MULTI AGENT - " + AGENT_NAME
else:
    AGENT_NAME = "SINGLE AGENT - " + AGENT_NAME

scores = ddpg(AGENT_NAME, multiple_agents = MULTI)
>>>>>>> f3a172e9d7954f8c7474f4d363a2cf7f4b85d97a

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()