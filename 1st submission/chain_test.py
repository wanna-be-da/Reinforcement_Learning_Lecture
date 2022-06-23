from chain_mdp import ChainMDP
from agent_chainMDP import agent
import numpy as np
from tempfile import TemporaryFile


# recieve 1 at rightmost state and recieve small reward at leftmost state
env = ChainMDP(10)
s = env.reset()

""" Your agent"""

#default parameters
sa_list = []

for i in range(env.n):
    for j in range(2):
        sa_list.append((i, j))

agent_params = {'gamma'            : 0.9,
                'kappa'            : 1.0,
                'mu0'              : 0.0,
                'lamda'            : 4.0,
                'alpha'            : 3.0,
                'beta'             : 3.0,
                'max_iter'         : 100,
                'sa_list'          : sa_list}

#initialize agent
agent = agent(agent_params)

# always move right left: 0, right: 1
# action = 1

#Code below is used for training the agent

def training(k):
   for episode in range(k):
       s = env.reset()
       done = False

       while not done:
           a = agent.take_action(s, 0)
           # Step environment
           s_, r, done, t = env.step(a)
           agent.observe([t, s, a, r, s_])
           agent.update_after_step(10, True)
           # Update current state
           s = s_ 
       #print(agent.pi)

#training for 1000 episodes
training(100)


cum_reward = 0.0
s = env.reset()
done = False

while not done: 
    action = agent.take_action(s, 0)
    s_, r, done, t = env.step(action)
    print(s_, r, t)
    cum_reward += r
    s = s_

print(f"total reward: {cum_reward}")
print(agent.pi)
print(type(agent.Ppost))
print(type(agent.Rpost))

#utfile = TemporaryFile()
np.save('./ppost', agent.Ppost)
np.save('./rpost', agent.Rpost)

print("this is")
rpost = np.load('./rpost.npy')
print(rpost)