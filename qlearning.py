import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human") # render mode should be human
env.reset()

high_obs_space = env.observation_space.high
low_obs_space = env.observation_space.low
num_actions = env.action_space.n

# how would we create our QTables

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (high_obs_space - low_obs_space) / DISCRETE_OBS_SIZE
# print(DISCRETE_OBS_SIZE[0])
# print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [num_actions]))
print(q_table.shape)

# done  = False
# # all our code should be wrapped into an environment i.e you should wrap your whole problem into an environment class
# # idea, create a library that can be used to wrap any problem into an environment, by formal MDP definition
# while not done:
#     action = 2 # action provided by the agent
#     state, reward, done, _ , _ = env.step(action) # step models the environment dynamics function p
#     print("The new state: ", state)
#     env.render()
    
# env.close()
    