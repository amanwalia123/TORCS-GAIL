from gym_torcs import TorcsEnv
import torch
import numpy as np
from actor import ActorNetwork
from torch.utils.tensorboard import SummaryWriter
from utility import calc_disc_return
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description="Play recorded expert episodes")
parser.add_argument('--file',     required=True,           help='Expert episode file (.pkl format)')
parser.add_argument('--num_play', required=True, type=int, help='Number of times to play episodes' )
my_args = parser.parse_args()

gamma = 0.95                          # discount factor
VISION = False                        # using torcs in measurement based mode (instead vision based mode) 


def load_expert_trajectory(filepath):
    episode = pickle.load(open(filepath,"rb"))
    states = episode['state']
    actions = episode['action']
    return states,actions

def play_trajectory(filepath,counts):
    
    # Defining torcs environment
    env = TorcsEnv(vision=VISION,throttle=True,gear_change=False)


    states,actions = load_expert_trajectory(filepath)
    
    for i in range(counts):
        print("Playing count : {}".format(i))
        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        for action in actions:
        
            # take action and observe reward and next state     
            ob, reward, done, info = env.step(action)

            if done:
                break


def main():
    # eps_file = "/home/aman/Programming/RL-Project/Expert_Trajectories/episode-751.pkl"
    # counts = 2
    play_trajectory(my_args.file,my_args.num_play)

if __name__ == "__main__":
   main()

