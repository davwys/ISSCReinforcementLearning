# Experiment 2:
# -------------
#
# We further optimize the DQN agent through parameter tuning to improve its performance.
#

import gym
import time
import tensorflow as tf

from stable_baselines.deepq.policies import CnnPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


# Agent class
class Agent:
    def __init__(self, name, model, algo):
        self.name = name
        self.model = model
        self.algo = algo


# Evaluation class
class Evaluation:
    def __init__(self, name, mean, std):
        self.name = name
        self.mean = mean
        self.std = std


# Create environment
env = gym.make('MsPacman-ram-v0')

# Disable deprecated logging on Tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

# Instantiate the agent
dqn_model = DQN(
    MlpPolicy, # TODO CnnPolicy
    env,
    buffer_size=10000,
    learning_rate=float(1e-4),
    learning_starts=10000,
    target_network_update_freq=1000,
    train_freq=4,
    exploration_final_eps=0.01,
    exploration_fraction=0.1,
    prioritized_replay_alpha=0.6,
    prioritized_replay=True,
    verbose=0
)

print('-----------------')

# Train all agents
print("Training DQN agent")

dqn_model.learn(total_timesteps=10000)  # TODO 10^7

# Save the agent
save_name = 'DQN_trained_v2'
print('Saving', save_name)
dqn_model.save(save_name)
print('')

# Vectorize environment
env = DummyVecEnv([lambda: env])

print('-----------------')

# Evaluate agent
print('Evaluating DQN...',)
mean_reward, std_reward = evaluate_policy(dqn_model, env, n_eval_episodes=50)
evaluation = Evaluation('DQN', mean_reward, std_reward)


print('Evaluation:')
print('-----------------')
print(evaluation.name, ':', evaluation.mean, ',', evaluation.std)

print('-----------------')

# Load the trained agent
best_model = DQN.load('DQN_trained_v2', env=env)

game_speed = 1  # 1 is equal to real-time, increase for faster playback

# View best trained agent
observation = env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        time.sleep(1/24/game_speed)
        action, _states = best_model.predict(observation)
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
