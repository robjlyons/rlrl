# Contrastive Variational Reinforcement Learning for Complex Observations 

This is a slightly tweaked version of Yusufma03 [CVRL](https://github.com/Yusufma03/CVRL)

This implementation uses dreamer version 1 as it's base, in ordser to get it to work with RetroArch I merged some env parameters of DreamerV2 in.

## How to run

- install requirements
```
pip install -r requirements.txt
```

- replace atari_preprocessing.py in gym\wrappers with the one in this folder.

- run cvrl.py with the following
```
python cvrl.py --logdir [DIRECTORY]/cvrl/1 --task atari_RetroArch
```
