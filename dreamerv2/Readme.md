# Dreamerv2

This is a slightly tweaked version of danijar's [Dreamerv2](https://github.com/danijar/dreamerv2)

## How to run

- install requirements
```
pip install -r requirements.txt
```

- replace atari_preprocessing.py in gym\wrappers with the one in this folder.

- run train.py with the following
```
python dreamerv2/train.py --logdir [DIRECTORY]/dreamerv2/1 --configs defaults atari --task atari_RetroArch
```

## Config file

The config file for dreamerv2 is very detailed and comprehensive. The only changes I have made are:

- dataset: {batch: 8, length: 50} - Lowered batch from 16 to 8 as my computer is a potato
- expl_behavior: Plan2Explore - Needed for more spare reward envs, you try 'greedy' which make work for Mario.
