import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0") # render mode should be human
env.reset()


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPISODES_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON/(END_EPISODES_DECAYING - START_EPSILON_DECAYING)

SHOW_EVERY = 200


high_obs_space = env.observation_space.high
low_obs_space = env.observation_space.low
num_actions = env.action_space.n

# how would we create our QTables

DISCRETE_OBS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (high_obs_space - low_obs_space) / DISCRETE_OBS_SIZE
# print(DISCRETE_OBS_SIZE[0])
# print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBS_SIZE + [num_actions]))
# print(q_table.shape)

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
# returns discete state from continous state
def quantization(state):
    discrete_state = (state - low_obs_space)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int64))

for episode in range(EPISODES):
    print(episode)
    episode_reward = 0
    render = False
    if episode % SHOW_EVERY == 0:
        print("*"*2)
        render = True
    initial_state = quantization(env.reset()[0])
    # print(initial_state)
    # print(np.argmax(q_table[initial_state]))
    done  = False
    # all our code should be wrapped into an environment i.e you should wrap your whole problem into an environment class
    # idea, create a library that can be used to wrap any problem into an environment, by formal MDP definition
    discrete_state = initial_state
    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(q_table[discrete_state]) # action provided by the agent
        else:
            # select random action
            action = np.random.randint(0, num_actions) # select random action from actions
        new_state, reward, done, _ , _ = env.step(action) # step models the environment dynamics function p
        episode_reward += reward
        new_discrete_state = quantization(new_state)
        
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = ((1 - LEARNING_RATE)*current_q) + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )]  = new_q
        
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
    
    if END_EPISODES_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= EPSILON_DECAY_VALUE
        
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        # update dictionary
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards["max"].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
    
                                      
    
        
env.close()



# # watch demo of the reinforcement learning agent
# env = gym.make("MountainCar-v0", render_mode="human") # render mode should be human
# env.reset()
# initial_state = quantization(env.reset()[0])
# done = False
# discrete_state = initial_state
# for i in range(1000):
#     while not done:
#         action = np.argmax(q_table[discrete_state])
#         new_state, reward, done, _, _ = env.step(action) # step models the environment dynamics
#         discrete_state = quantization(new_state)
#         env.render()

# env.close()

print(aggr_ep_rewards)

plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["avg"], label="avg")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["min"], label="min")
plt.plot(aggr_ep_rewards["ep"], aggr_ep_rewards["max"], label="max")

plt.legend(loc=4)
plt.show()
