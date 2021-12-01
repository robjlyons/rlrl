# Stable Baselines 2

I suggest having a read through their docs [stablebaselines](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

This package allows several algorithms to be used with games:
-A2C
-ACER
-ACKTR
-DQN
-HER
-GAIL
-PPO1
-PPO2

I have also included a version that mixes an algorithm with a genetic system like NEAT.

## How To Run

- install requirements
```
pip install -r requirements.txt
```
- make sure RetroArch is open and on the menu screen
- run PPO2.py
```
python PPO2.py
```

## Explanation

### Imports

```
from stable_baselines.common.cmd_util import make_vec_env - For observation
from stable_baselines.common.vec_env import VecFrameStack - Stacks the frames so the agent can see movement
from stable_baselines import PPO2 - Which algorithm to import, change to use another algorithm
```

### Environment

should be fairly self explanatory
```
# Define environment
env = make_vec_env('RetroArch-v0')
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)
```

### Model

You can change 'PPO2' to another imported algorithm. 'CnnPolicy' means the agent is using images for training.
```
model = PPO2('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000) - Change timesteps to train for longer or shorter, be aware than the benchmarks for these algorithms train for 100 million steps or more.
```

### Test

Tests the trained agent until closed.
```
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

### Save

Save the model for future use.
```
# Save model
model.save("ppo_GB")
```
