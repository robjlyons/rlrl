from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.ppo import PPO

# Environment
env = make_vec_env('RetroArch-v0')
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

# Training
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("ppo_mario_GB")