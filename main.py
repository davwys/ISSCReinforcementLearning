import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import ACKTR
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR, A2C, DDPG
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Create environment
env = gym.make('BipedalWalker-v3')

# TODO learning rate

# Instantiate the agent
ppo2_model = PPO2(MlpPolicy, env, learning_rate=2.5e-4, verbose=1)
a2c_model = A2C(MlpPolicy, env, learning_rate=2.5e-4, verbose=1)
acktr_model = ACKTR(MlpPolicy, env, learning_rate=2.5e-4, verbose=1)
ddpg_model = DDPG(MlpPolicy, env, learning_rate=2.5e-4, verbose=1)

# Create array of objects with model & name
models = [
    {'name': 'ppo2', 'model': ppo2_model},
    {'name': 'a2c', 'model': a2c_model},
    {'name': 'acktr', 'model': acktr_model},
    {'name': 'ddpg', 'model': ddpg_model},
]

# TODO total_timesteps=int(2e5)

# Train all agents
for model_object in models:
    model_object.model.learn(
        total_timesteps=10000
    )
    #
    # Save the agent
    print("Saving!")
    model_object.model.save("ppo2_walker")


# Evaluate agents
evaluations = []
for model_object in models:
    mean_reward, std_reward = evaluate_policy(model_object.model, env, n_eval_episodes=100)
    evaluations.append({
        'name': model_object.name,
        'mean': mean_reward,
        'std': std_reward
    })

# Vectorize environment
env = DummyVecEnv([lambda: env])

# Load the trained agent
model = PPO2.load("ppo2_walker", env=env)

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
env.close()