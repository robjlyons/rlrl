from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2

# Define environment
env = make_vec_env('RetroArch-v0')
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = PPO2('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)


obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Save model
model.save("ppo_GB")
