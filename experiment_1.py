import gym
import time
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DeepQMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, ACKTR, A2C, DQN
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

# Disable deprecated logging on tensorflow
tf.logging.set_verbosity(tf.logging.ERROR)

# Instantiate the agent
ppo2_model = PPO2(MlpPolicy, env, verbose=0)
a2c_model = A2C(MlpPolicy, env, verbose=0)
acktr_model = ACKTR(MlpPolicy, env, verbose=0)
dqn_model = DQN(DeepQMlpPolicy, env, verbose=0)

# Create array of Agents with model, name & algorithm
agents = [
    Agent('PPO2', ppo2_model, PPO2),        # Proximal Policy Optimization
    Agent('A2C', a2c_model, A2C),           # Synchronous Advantage Actor Critic
    Agent('ACKTR', acktr_model, ACKTR),     # Actor Critic using Kronecker-Factored Trust Region
    Agent('DQN', dqn_model, DQN),           # Deep Q Network
]

print('-----------------')

# Train all agents
for agent in agents:
    print("Training model", agent.name)

    agent.model.learn(
        total_timesteps=100000
    )
    # Save the agent
    save_name = agent.name + '_trained'
    print('Saving', save_name)
    agent.model.save(save_name)
    print('')

# Vectorize environment
env = DummyVecEnv([lambda: env])

print('-----------------')

# Evaluate agents
evaluations = []
highest_mean = 0
best_agent = None

for agent in agents:
    print('Evaluating', agent.name)
    mean_reward, std_reward = evaluate_policy(agent.model, env, n_eval_episodes=50)
    evaluations.append(Evaluation(agent.name, mean_reward, std_reward))

    # Determine if this was the highest-scoring agent yet
    if mean_reward > highest_mean:
        best_agent = agent

print('Final evaluation:')
print('-----------------')
for evaluation in evaluations:
    print(evaluation.name, ':', evaluation.mean, ',', evaluation.std)

print('Best agent:', best_agent.name)

# Load the best trained agent
best_agent_name = best_agent.name + '_trained'
best_model = best_agent.algo.load(best_agent_name, env=env)

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
