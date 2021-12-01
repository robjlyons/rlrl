# Stable Baselines 3

I suggest having a read through their docs [stablebaselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

Stable Baselines 3 is pretty much the same as the previous version with two major differences
1. It's slower, your mileage may vary but using torch-1.7.1+cu110 I saw a loss of about 10 fps
2. It's neing updated. This is where the new algorithms will be added.

This package allows several algorithms to be used with games:
- A2C
- DQN
- HER
- PPO
- QR-DQN
- Maskable PPO

## How To Run

- install requirements
```
pip install -r requirements.txt
```
- You will need to install the correct version of pytorch for your cuda version
- make sure RetroArch is open and on the menu screen
- run PPO.py
```
python PPO.py
```

## Explanation

### Imports

```
from stable_baselines3.common.env_util import make_vec_env - For observation
from stable_baselines3.common.vec_env - Stacks the frames so the agent can see movement
from stable_baselines3.ppo import PPO - Which algorithm to import, change to use another algorithm
```

### Environment

should be fairly self explanatory
```
# Environment
env = make_vec_env('RetroArch-v0')
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)
```

### Model

You can change 'PPO' to another imported algorithm. 'CnnPolicy' means the agent is using images for training.
```
# Training
model = PPO('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000) - Change timesteps to train for longer or shorter, be aware than the benchmarks for these algorithms train for 100 million steps or more.
```

### Save

Save the model for future use.
```
# Save model
model.save("ppo_mario_GB")
```
