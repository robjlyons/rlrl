# rlrl
Real Life Reinforcement Learning - Multiple algorithm implementation on RetroArch Gameboy Super Mario Land

Implementations of RL (and others) algorithms on a real time game. I'm using the gameboy mario game runing on RetroArch but essentially any game can be used with 'some' tweaking.

Where gym atari envs use ram to track scores and so on, here I am using just computer vision.

I strongly suggest using a conda environment for all of this, prefferably for each algorithm as they can use different libraries, especially things like torch and tensorflow.

## Caution

These implementations are not tested to convergence. I own a potato as a PC and therefore have only been able to test upto 1m frames on very low batch sizes..

## Implementations TODO

- [x] [NEAT](neat/Readme.md)
- [x] [Stable Baselines 2](stable_baselines/Readme.md)
- [x] [Stable Baselines 3](stable_baselines_3/Readme.md)
- [x] [DreamerV2](dreamerv2/Readme.md)
- [x] [Intrinsic Curiosity Module](ICM/Readme.md)
- [x] [Random Network Distillation](RND/Readme.md)
- [x] [Contrastive Variational Model-Based Reinforcement Learning for Complex Observations](CVRL/Readme.md)
