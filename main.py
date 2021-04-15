import gym

import numpy as np
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR, A2C
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Agent class
class Agent:
    def __init__(self, name, model):
        self.name = name
        self.model = model

# Evaluation class
class Evaluation:
    def __init__(self, name, mean, std):
        self.name = name
        self.mean = mean
        self.std = std


# Create environment
env = gym.make('BipedalWalker-v3')

# Disable deprecated logging on tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

# TODO learning rate

# Instantiate the agent
ppo2_model = PPO2(MlpPolicy, env, learning_rate=2.5e-4, verbose=0)
a2c_model = A2C(MlpPolicy, env, learning_rate=2.5e-4, verbose=0)
acktr_model = ACKTR(MlpPolicy, env, learning_rate=2.5e-4, verbose=0)
# ddpg_model = DDPG(MlpPolicy, env, learning_rate=2.5e-4, verbose=1) TODO

# Create array of objects with model & name
models = [
    Agent('PPO2', ppo2_model),
    Agent('A2C', a2c_model),
    Agent('ACKTR', acktr_model),
]

print('-----------------')

# Train all agents
for model_object in models:
    print("Training model", model_object.name)

    # TODO total_timesteps=int(2e5)
    model_object.model.learn(
        total_timesteps=int(1e5)
    )
    # Save the agent
    save_name = model_object.name + '_walker'
    print("Saving", save_name)
    model_object.model.save(save_name)

# Vectorize environment
env = DummyVecEnv([lambda: env])

print('-----------------')

# Evaluate agents
evaluations = []
for model_object in models:
    print("Evaluating", model_object.name)
    mean_reward, std_reward = evaluate_policy(model_object.model, env, n_eval_episodes=10)
    evaluations.append(Evaluation(model_object.name, mean_reward, std_reward))

print('Final evaluation:')
print('-----------------')
for evaluation in evaluations:
    print(evaluation.name, ":", evaluation.mean, ",", evaluation.std)

#
# # Load the trained agent
# model = PPO2.load("ppo2_walker", env=env)
#
# # Evaluate the agent
# print("Eval:")
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#
# # Enjoy trained agent
# observation = env.reset()
# print("Running!")
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
env.close()