import time
import gym
import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.callbacks import CheckpointCallback

def mutate(params):
    """Mutate parameters by adding normal noise to them"""
    return dict((name, param + np.random.normal(size=param.shape))
                for name, param in params.items())

def evaluate(env, model):
    """Return mean fitness (sum of episodic rewards) for given model"""
    episode_rewards = []
    for _ in range(1):
        current_max_fitness = 0
        frame = 0
        counter = 0
        fitness = 0.0
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            fitness += reward

            if fitness > current_max_fitness:
                current_max_fitness = fitness
                counter = 0
            else:
                counter += 1
                
            if done or counter == 250:
                done = True

        episode_rewards.append(fitness)
    return np.mean(episode_rewards)

# Retro Environment
env = make_vec_env("RetroArch-v0")
env = VecFrameStack(env, n_stack=4)


model = PPO2('CnnPolicy', env, ent_coef=0.01, learning_rate=0.001, verbose=1, policy_kwargs={'net_arch': [8, ]})
# Use traditional actor-critic policy gradient updates to
# find good initial parameters
model.learn(total_timesteps=100000)


#model = PPO2.load("PPO2_Gen_model.zip", env)


# Get the parameters as the starting point for ES
mean_params = model.get_parameters()

# Include only variables with "/pi/" (policy) or "/shared" (shared layers)
# in their name: Only these ones affect the action.
mean_params = dict((key, value) for key, value in mean_params.items()
                   if ("/pi/" in key or "/shared" in key))

for iteration in range(1000):
    # Create population of candidates and evaluate them
    population = []
    for population_i in range(20):
        candidate = mutate(mean_params)
        # Load new policy parameters to agent.
        # Tell function that it should only update parameters
        # we give it (policy parameters)
        model.load_parameters(candidate, exact_match=False)
        model.save("A2C_Gen_Pop")
        fitness = evaluate(env, model)
        population.append((candidate, fitness))
        print('Num Population:', population_i)
        print('Pop Score:', fitness)
    # Take top 10% and use average over their parameters as next mean parameter
    top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:10]
    mean_params = dict(
        (name, np.stack([top_candidate[0][name] for top_candidate in top_candidates]).mean(0))
        for name in mean_params.keys()
    )
    mean_fitness = sum(top_candidate[1] for top_candidate in top_candidates) / 10.0
    print("Iteration {:<3} Mean top fitness: {:.2f}".format(iteration, mean_fitness))
    model.save("PPO2_Gen_model")