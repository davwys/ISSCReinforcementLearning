import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLander-v2')
#
# # Instantiate the agent
# model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
# # Train the agent
# print("Training!")
# model.learn(total_timesteps=int(2e5))
#
# # Save the agent
# print("Saving!")
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
print("Eval:")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
observation = env.reset()
print("Running!")
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

# Create Environment
# env = gym.make('BipedalWalker-v3')
# env.reset()
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# Instantiate the agent
# model = PPO2(
#     'MlpPolicy',
#     env,
#     normalize=true,
#     n_envs=16,
#     n_timesteps=!!float 5e6,
#     n_steps=2048,
#      nminibatches=32,
#     lam=0.95,
#     gamma=0.99,
#     noptepochs=10,
#     ent_coef=0.001,
#     learning_rate=!!float 2.5e-4,
#     cliprange=0.2,
#     verbose=1
# )
#
# # Train the agent
# model.learn(total_timesteps=10000)
#
# # Save the agent
# model.save("dqn_lunar")
# del model  # delete trained model to demonstrate loading
#
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(1000):
#         action, _states = model.predict(observation)
#         observation, reward, done, info = env.step(action)
#         env.render()
#
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
# env.close()