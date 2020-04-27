import torch
import os
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
#import ipdb

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from EpisodeSaver import *

state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  # to change
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = True    # train or not
TAU = 0.001
MAX_STEPS = 50000

VISION = False

TRAJECTORY_DIR = "/home/aman/Programming/DDPG_Torcs_PyTorch/expert_trajectories"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
OU = OU()


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)


actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

############# Choose which trained model to load ############
ALPINE = True                                               #
CG = False                                                  #
#############################################################

try:
    if ALPINE:
        actor.load_state_dict(torch.load('actormodel-alpine.pth'))
        critic.load_state_dict(torch.load('criticmodel-alpine.pth'))
    if CG:
        actor.load_state_dict(torch.load('actormodel-cg.pth'))
        critic.load_state_dict(torch.load('criticmodel-cg.pth'))
    actor.eval()
    critic.eval()
    print("model load successfully")
except:
    print("cannot find the model")

#critic.apply(init_weights)
buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum')

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

#env environment
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

for i in range(0, 1500):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    if (i % 5 == 0):
        train_indicator = False
        print(" Episode {} ,Testing".format(i))
    else:
        train_indicator = True
        print("Episode {} ,Training".format(i))

    saver = EpiosdeSaver(os.path.join(TRAJECTORY_DIR, "episode-{}.pkl".format(i+1)),
                         ['state', 'action', 'reward', 'next_state', 'done'])

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                     ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    for j in range(MAX_STEPS):
        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])
        #ipdb.set_trace()
        a_t_original = actor(torch.tensor(s_t.reshape(
            1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()
        #print(type(a_t_original[0][0]))

        noise_t[0][0] = train_indicator * \
            max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * \
            max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * \
            max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        # stochastic brake
        if random.random() <= 0.1:
            # print("apply the brake")
            noise_t[0][2] = train_indicator * \
                max(epsilon, 0) * \
                OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)

        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0])

        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX,
                          ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        #add to replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = buff.getBatch(BATCH_SIZE)

        # torch.cat(batch[0])
        states = torch.tensor(np.asarray(
            [e[0] for e in batch]), device=device).float()
        actions = torch.tensor(np.asarray(
            [e[1] for e in batch]), device=device).float()
        rewards = torch.tensor(np.asarray(
            [e[2] for e in batch]), device=device).float()
        new_states = torch.tensor(np.asarray(
            [e[3] for e in batch]), device=device).float()
        dones = np.asarray([e[4] for e in batch])
        y_t = torch.tensor(np.asarray([e[1]
                                       for e in batch]), device=device).float()

        #use target network to calculate target_q_value
        target_q_values = target_critic(new_states, target_actor(new_states))

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if(train_indicator):

            #training
            q_values = critic(states, actions)
            loss = criterion_critic(y_t, q_values)
            optimizer_critic.zero_grad()
            # for param in critic.parameters(): param.grad.data.clamp(-1, 1)
            loss.backward(retain_graph=True)
            optimizer_critic.step()

            a_for_grad = actor(states)
            a_for_grad.requires_grad_()  # enables the requires_grad of a_for_grad
            q_values_for_grad = critic(states, a_for_grad)
            critic.zero_grad()
            q_sum = q_values_for_grad.sum()
            q_sum.backward(retain_graph=True)

            grads = torch.autograd.grad(q_sum, a_for_grad)

            act = actor(states)
            actor.zero_grad()
            act.backward(-grads[0])
            optimizer_actor.step()

            #soft update for target network
            #actor_params = list(actor.parameters())
            #critic_params = list(critic.parameters())
            # print("soft updates target network")
            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict(
                )[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict(
                )[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)

        # print("---Episode ", i , "state " , s_t, "  Action:", a_t[0], "  Reward:", r_t, "  Loss:", loss)

        if j >= MAX_STEPS - 1:
            done = True

        # Saving the episode
        saver.add([s_t, a_t[0], r_t, s_t1, done])

        if done and not train_indicator:

            print("Reached {} steps".format(j))
            saver.save_file()

        s_t = s_t1

        if done:
            break

    if np.mod(i, 3) == 0:
        if (train_indicator):
            print("saving model")
            torch.save(actor.state_dict(), 'actormodel-alpine.pth')
            torch.save(optimizer_actor.state_dict(),'optim_actor-alpine.pth')

            torch.save(critic.state_dict(), 'criticmodel-alpine.pth')
            torch.save(optimizer_critic.state_dict(),'optim_critic-alpine.pth')



env.end()
print("Finish.")

#for param in critic.parameters(): param.grad.data.clamp(-1, 1)
