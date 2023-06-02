import gymnasium as gym
# from stable_baselines3 import A2C
# import numpy as np
from video_gen import generate_video_from_numpy_array

# env = gym.make("LunarLander-v2", render_mode="rgb_array")

# #  define model
# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000)

# renders = []
# episodes = 3

# for ep in range(episodes):
#     (obs, _) = env.reset()
    
#     while True:
#         action = model.predict(obs, deterministic=True)
#         new_obs, reward, terminated, truncated, _  = env.step(action)
        
#         renders.append(env.render())
        
#         if terminated or truncated:
#             break

# renders = np.asarray(renders)

# import gym

from stable_baselines3 import DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v2")

# Instantiate the agent
model = A2C("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("a2c_lunar")
del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = A2C.load("a2c_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
renders = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    renders.append(vec_env.render("rgb_array"))

# generate and save video
generate_video_from_numpy_array(renders, 
                                height=renders[0].shape[0], 
                                width=renders[0].shape[1], 
                                outfile_name="lunarlander.mp4")
