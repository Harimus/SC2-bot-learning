import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import matplotlib.pyplot as plt
import VPG
from policy import ActorCriticPolicy
#expects you to have spinup installed with pip install -e
import spinup


if __name__ == '__main__':
    steps_per_epoch=4000
    epochs=50
    gamma=0.99
    pi_lr=3e-4
    vf_lr=1e-3
    train_v_iters=80
    lam=0.97
    spinupVPG = spinup.vpg_pytorch(lambda: gym.make('MountainCar-v0'), epochs=20)

    print('++++++++++++++++++FROM HERE MY CODE++++++++++++++++++++++')
    myVPG = spinup.vpg_pytorch(lambda: gym.make('MountainCar-v0'), actor_critic=ActorCriticPolicy, epochs=20)



