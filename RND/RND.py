import datetime
from time import time

import numpy as np

import torch as T
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env, make_vec_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.running_mean_std import RunningMeanStd

from tqdm import trange
from itertools import count
# from google.colab import drive


ENV_NAME = 'RetroArch-v0'
SAVE_PATH = './'

TOTAL_FRAMES = 1e7    # 5 million frames
ROLLOUT_LENGTH = 128  # transitions in each rollout
NENV = 1              # parallel environments, increase to decorrolate batches
GAMMA = 0.99          # reward discounting coefficient
LAMBDA = 0.95         # GAE
SEED = 420            # blaze it
MB_SPLIT = 4          # split minibatch into quarters for surrogate calculation
MB_EPOCHS = 4         # epochs per minibatch (4 weight updates on AC per mb)
CLIP = 0.2            # clip the surrogate objective so updates are not too large

STEPS_PER_ROLLOUT = ROLLOUT_LENGTH*NENV
TOTAL_UPDATES = int(TOTAL_FRAMES // STEPS_PER_ROLLOUT)
SPLIT_LEN = int(STEPS_PER_ROLLOUT//MB_SPLIT)
DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
print(DEVICE, T.version.cuda)

set_global_seeds(SEED)


def conv_size(net, in_shape):
    """ util for calculating flat output shape of a given net """
    x = Variable(T.rand(1, *in_shape))
    o = net(x)
    b = (-1, o.size(1), o.size(2), o.size(3))
    return b, o.data.view(1, -1).size(1)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.01)
        elif 'weight' in name:
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')

class ACC(nn.Module):
    """ RND Uses two value heads/critics. One criticises the extrinsic reward
        prediction. The other criticises the intrinsic reward prediction.
    """
    def __init__(self, input_shape, num_actions):
        super().__init__()
        h, w, c = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
        )
        conv_x, conv_f = conv_size(self.conv, (c,h,w))
        self.conv = nn.Sequential(
            self.conv,
            nn.Flatten(),
            nn.Linear(conv_f, 512),
            nn.ReLU(True)
        )
        self.actor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, num_actions)
        )
        self.ext_critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        self.int_critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )
        
        self.apply(init_weights)

    def forward(self, x):
        latent = self.conv(x)
        return self.actor(latent), self.ext_critic(latent), self.int_critic(latent)


class RND(nn.Module):
    """ RND sources its intrinsic bonus from the ability of a predictor
        network to estimate the output of a fixed network, where the 
        predictor is only trained by observing the 512-dim output of the
        fixed. The fixed network cannot adapt to the scale of it's inputs,
        so, as mentioned in the paper (2.4), it is crucial to normalize its
        observations/inputs.
    """
    def __init__(self, input_shape):
        super().__init__()
        
        h, w, c = input_shape

        self.fixed_conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True)
        )
        _, f = conv_size(self.fixed_conv, (c,h,w))
        self.fixed_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f, 512),
        )
        self.fixed=nn.Sequential(
            self.fixed_conv, 
            self.fixed_linear
        )
        for param in self.fixed.parameters():
            param.requires_grad = False
            

        self.predictor_conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True)
        )
        _, f = conv_size(self.predictor_conv, (c,h,w))
        self.predictor_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f, 512),
        )
        self.predictor=nn.Sequential(
            self.predictor_conv, 
            self.predictor_linear
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.fixed(x), self.predictor(x)



class Logger:

    def __init__(self, print_rate=250):
        self.log = {'ep_r':[], 'ep_l':[], 'loss':[], 'pgloss':[], 
                    'vloss':[], 'ent':[], 'int':[]}
        self.n_ep = 0              # total games/episodes
        self.n_update = 1          # total weight updates
        self.n_frames = 0          # env steps (total from checkpoint)
        self.run_frames = 0        # env steps (for this run)
        self.max_rwd = -np.inf     # max rwd out of all games played
        self.start_time = time()   # time we started *this* run
        self.last_checkpoint = 0   # total_frames at last checkpoint
        self.print_rate = print_rate

    def eta(self):  # hh:mm:ss left to train
        elapsed_time = time() - self.start_time
        frames_left = TOTAL_FRAMES - self.n_frames
        sec_per_frame = elapsed_time / self.n_frames
        sec_left = int(frames_left * sec_per_frame)
        eta_str = str(datetime.timedelta(seconds=sec_left))
        return eta_str

    def fps(self):  # frames per second
        elapsed_time = time() - self.start_time
        fps = int(self.run_frames / elapsed_time)
        return fps

    def sma(self, x):  # simple moving average
        if len(x) == 0: return 'NaN'
        div = 200 if len(x) > 200 else len(x)
        return sum(list(zip(*x[-div:]))[-1])/div

    def print_log(self):
        fps = self.fps()
        eta = self.eta()
        print('-'*10, self.n_update, '/', TOTAL_UPDATES, '-'*10)
        print('Num Games:', self.n_ep)
        print('Num Frames:', self.n_frames)
        print('FPS:', fps)
        print('ETA:', eta)
        print('SMA Length:', self.sma(self.log['ep_l']))
        print('SMA Reward:', self.sma(self.log['ep_r']))
        print('SMA Intrinsic Reward:', self.sma(self.log['int']))
        print('SMA Entropy:', self.sma(self.log['ent']))
        print('SMA Loss:', self.sma(self.log['loss']))
        print('SMA PG Loss:', self.sma(self.log['pgloss']))
        print('SMA V Loss:', self.sma(self.log['vloss']))
        print('Max reward:', self.max_rwd)

    def record(self, ep, loss, pgloss, vloss, ent, intr):
        
        self.n_update += 1
        self.n_frames += STEPS_PER_ROLLOUT
        self.run_frames += STEPS_PER_ROLLOUT
        fr = (self.n_frames, self.n_update)

        # stats about finished episodes/games
        for l, r in zip(ep['l'], ep['r']):
            self.log['ep_l'].append(fr+(l,))
            self.log['ep_r'].append(fr+(r,))
            if r > self.max_rwd: self.max_rwd = r
            self.n_ep += 1
             
        # nn training statistics
        self.log['loss'].append(fr+(loss,))
        self.log['pgloss'].append(fr+(pgloss,))
        self.log['vloss'].append(fr+(vloss,))
        self.log['ent'].append(fr+(ent,))
        self.log['int'].append(fr+(intr,))
        
        # print log
        if self.n_update % self.print_rate == 0:
            self.print_log()


def ob_to_torch(x):
    x = np.moveaxis(x, -1, 1)
    x = T.from_numpy(x).float()
    x = x.to(DEVICE)
    return x



def rollout_generator(env, acc):
    """ :param acc: ActorCriticCritic policy CNN """

    ob = ob_to_torch(env.reset())
    mb = {'obs':[], 'nobs':[], 'act':[], 'logp':[], 'ext_rwd':[], 
          'done':[], 'int_v':[], 'ext_v':[]}
    ep = {'l':[], 'r':[]}  # len & total reward of done eps 
    nv = {'n_ext_v':0, 'n_int_v':0}  # next values for GAE 
    
    for step in count(1):

        # we compute gradients in the train loop
        with T.no_grad():
            logits, ext_v, int_v = acc(ob)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        new_ob, ext_rwd, done, info = env.step(action)
        
        # store transition
        mb['obs'].append(ob)
        mb['act'].append(action)
        mb['logp'].append(logp)
        mb['ext_rwd'].append(T.from_numpy(ext_rwd))
        mb['int_v'].append(int_v.view(-1))
        mb['ext_v'].append(ext_v.view(-1))
        mb['done'].append(T.from_numpy(done))
        for t in info:
            if t.get('episode'):
                ep['l'].append(t['episode']['l'])
                ep['r'].append(t['episode']['r'])

        ob = ob_to_torch(new_ob)

        if step % ROLLOUT_LENGTH != 0:
            continue

        # intrinsic reward is computed on S(t+1) (next obs)
        mb['nobs'] = mb['obs'][1:] + [ob]

        # make batch tensor
        for k in mb.keys():
            mb[k] = T.stack(mb[k]).to(DEVICE)
        
        # next values for GAE
        with T.no_grad(): 
            _, ext_v, int_v = acc(ob)
            nv['ext_v'] = ext_v.view(-1)
            nv['int_v'] = int_v.view(-1)

        yield mb, nv, ep
        mb = {'obs':[], 'nobs':[], 'act':[], 'logp':[], 'ext_rwd':[], 
              'done':[], 'int_v':[], 'ext_v':[]}
        ep = {'l':[], 'r':[]}
        nv = {'ext_v':0, 'int_v':0}


def new_run():
    """ setup parallelised gym environments (MaxAndSkip=4 by default) """
    env = make_atari_env(ENV_NAME, num_env=NENV, seed=SEED)
    env._max_episode_steps = 4500*4
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_reward=False, clip_reward=1e5, gamma=1.0)
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n

    # observations are normalized, as per paper reccomendation.
    print('Initializing obs normalizer...')
    ob = env.reset()
    init_obs = [ob]
    for i in trange(300):
        parallel_actions = np.random.randint(0, policy_dim, size=(NENV,))
        ob, *_ = env.step(parallel_actions)
    env.reset()

    # policy/actor + extrinsic critic + intrinsic critic
    acc_network = ACC(in_dim, policy_dim).to(DEVICE)
    acc_optimizer = Adam(acc_network.parameters(), 7e-4, eps=1e-5)

    # RND target + RND predictor
    rnd_network = RND(in_dim).to(DEVICE)
    rnd_optimizer = Adam(rnd_network.parameters(), 1e-4, eps=1e-5)

    # intrinsic reward standard deviation
    rnd_running_stats = RunningMeanStd()
    logger = Logger(print_rate=5)

    return env, acc_network, acc_optimizer, rnd_network, rnd_optimizer, rnd_running_stats, logger


def save_checkpoint(checkpoint_id):
    checkpoint = {
        'env': env, # save VecNormalized env mean and std
        'acc': acc_network.state_dict(),
        'rnd': rnd_network.state_dict(),
        'acc_opt': acc_optimizer.state_dict(),
        'rnd_opt': rnd_optimizer.state_dict(),
        'rnd_stat': rnd_running_stats,
        'logger': logger
    }
    fn = SAVE_PATH+str(checkpoint_id)+ENV_NAME+'.checkpoint.p'
    T.save(checkpoint, fn)

def load_checkpoint(file_name):
    checkpoint = T.load(file_name, map_location=T.device('cpu'))
    
    venv = make_atari_env(ENV_NAME, num_env=NENV, seed=SEED)
    venv = VecFrameStack(venv, n_stack=4)
    env = checkpoint['env']
    env.set_venv(venv)
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n

    acc_network = ACC(in_dim, policy_dim).to(DEVICE)
    acc_network.load_state_dict(checkpoint['acc'])
    acc_optimizer = Adam(acc_network.parameters())
    acc_optimizer.load_state_dict(checkpoint['acc_opt'])

    rnd_network = RND(in_dim).to(DEVICE)
    rnd_network.load_state_dict(checkpoint['rnd'])
    rnd_optimizer = Adam(rnd_network.parameters())
    rnd_optimizer.load_state_dict(checkpoint['rnd_opt'])

    rnd_running_stats = checkpoint['rnd_stat']
    logger = checkpoint['logger']

    return env, acc_network, acc_optimizer, rnd_network, rnd_optimizer, rnd_running_stats, logger

def GAE(rwds, vals, next_val, dones=None):
    """ GAE lambda. vals is full batch, next_val is singular """
    returns = T.empty_like(rwds).to(DEVICE)
    if dones is None: dones = T.ones_like(rwds, dtype=T.bool)
    g = 0
    for i in reversed(range(ROLLOUT_LENGTH)):
        d = rwds[i] + (GAMMA * next_val * ~dones[i]) - vals[i]
        g = d + (GAMMA * LAMBDA * ~dones[i] * g)
        returns[i] = g + vals[i]
        next_val = vals[i]
    adv = returns - vals
    return returns, adv


def distill(obs):
    """ distill the rnd network and return intrinsic rewards"""
    fixed_RND, predict_RND = rnd_network.forward(obs)
    RND_L2 = (fixed_RND - predict_RND).pow(2)
    
    # one intrinsic reward for each transition
    int_rwd = RND_L2.mean(1).detach()

    # distill RND network
    rnd_optimizer.zero_grad()
    loss = RND_L2.mean()
    loss.backward()
    nn.utils.clip_grad_norm_(rnd_network.parameters(), 0.5)
    rnd_optimizer.step()

    # update rnd normalization stats
    rnd_running_stats.update(int_rwd.cpu().numpy())
    int_rwd /= np.sqrt(rnd_running_stats.var) + 1e-7
    
    return int_rwd







#env, acc_network, acc_optimizer, rnd_network, rnd_optimizer, rnd_running_stats, logger = load_checkpoint(SAVE_PATH+'[SAVE FILE].p')
env, acc_network, acc_optimizer, rnd_network, rnd_optimizer, rnd_running_stats, logger = new_run()








rollouts = rollout_generator(env, acc_network)
scheduler = LambdaLR(acc_optimizer, lambda i: max(0.1, 1 - i/TOTAL_UPDATES))

for i_update in range(logger.n_update, TOTAL_UPDATES):

    mb, nv, ep = next(rollouts)
    
    # distill/train RND network and get intrinsic rewards
    s = mb['nobs'].size()
    mb['int_rwd'] = distill(mb['nobs'].view(STEPS_PER_ROLLOUT, *s[2:]))
    mb['int_rwd'] = mb['int_rwd'].view((ROLLOUT_LENGTH, NENV))  # for gae

    # get returns and generalized advantage estimates
    mb['ext_return'], mb['ext_gae'] = GAE(mb['ext_rwd'], mb['ext_v'], 
                                          nv['ext_v'], mb['done'])
    mb['int_return'], mb['int_gae'] = GAE(mb['int_rwd'], mb['int_v'], 
                                          nv['int_v'])
    
    # flatten to (NENV*ROLLOUT_LENGTH, ...)
    for k in mb.keys():
        s = mb[k].size()
        mb[k] = mb[k].view(STEPS_PER_ROLLOUT, *s[2:])

    # randomly split mb for surrogate calculation
    ixs = np.arange(STEPS_PER_ROLLOUT)
    np.random.shuffle(ixs)
    
    for epoch in range(MB_EPOCHS): # multiple updates using each minibatch
        for s in range(MB_SPLIT):  

            # split minibatch into chunks
            ix = ixs[s*SPLIT_LEN:s*SPLIT_LEN+SPLIT_LEN]
            s_obs = mb['obs'][ix]
            s_logp_old = mb['logp'][ix]
            s_act = mb['act'][ix]
            s_ext_return = mb['ext_return'][ix]
            s_int_return = mb['int_return'][ix]
            s_ext_v_old = mb['ext_v'][ix]
            s_int_v_old = mb['int_v'][ix]
            s_ext_adv = mb['ext_gae'][ix]
            s_int_adv = mb['int_gae'][ix]

            # policy gradient + entropy under new policy (same policy for s==1)
            logits, ext_v, int_v = acc_network(s_obs)
            dist = Categorical(logits=logits)
            s_ent = dist.entropy().mean()
            s_logp_new = dist.log_prob(s_act)

            # combine generalized advantages and normalize
            s_adv = (s_ext_adv + s_int_adv*0.5).detach()
            s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-7)

            # surrogate loss (ratio == 1 on first iter, ln cancels)
            ratio = (s_logp_new - s_logp_old).exp() # e^x / e^x == e^(x-y)
            L_CPI = s_adv * ratio
            L_CLAMP = s_adv * (ratio.clamp(1-CLIP, 1+CLIP))
            L_CLIP  = -T.min(L_CPI, L_CLAMP).mean()

            # value losses
            ext_V_L = (ext_v.view(-1) - s_ext_v_old).pow(2).mean()
            int_V_L = (int_v.view(-1) - s_int_v_old).pow(2).mean()
            V_L = ext_V_L + int_V_L
                    
            # step
            loss = L_CLIP + 0.5*V_L - 0.001*s_ent
            acc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(acc_network.parameters(), 0.5)
            acc_optimizer.step()


    # log & make a checkpoint every 1m frames
    scheduler.step(i_update)
    logger.record(ep, loss.item(), L_CLIP.item(), 
                  V_L.item(), s_ent.item(), 
                  s_int_return.mean().item())
    
    if logger.n_frames - logger.last_checkpoint > 1e5:
        save_checkpoint(logger.n_frames)
        logger.last_checkpoint = logger.n_frames

save_checkpoint('FINAL')





