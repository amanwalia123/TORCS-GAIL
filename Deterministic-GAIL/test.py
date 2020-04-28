from gym_torcs import TorcsEnv
import torch
import numpy as np
from actor import ActorNetwork
from torch.utils.tensorboard import SummaryWriter
from utility import calc_disc_return
import os

#######################################
#                                     #  
#          Hyperparameters            #  
#                                     #   
#######################################

random_seed = 100                     # random seed for experiment  
gamma = 0.95                          # discount factor
test_episodes = 100                   # number of evaluation episodes for validation reward
VISION = False                        # using torcs in measurement based mode (instead vision based mode)  
max_steps = 50000                     # maximum time steps for each episode  

weights_dir = "./weights"             # directory to store weights
logs_dir    = "./logs"                # directory to store tensorboard logs  


# Defining torcs environment
env = TorcsEnv(vision=VISION,throttle=True,gear_change=False)
state_size = 29                       # number of parameters in state space
action_size = 3                       # number of parameters in action space

exp_num = 1                           # experiment number

# Define the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected {} device".format(device))

# Defining the policy agent
actor = ActorNetwork(state_size).to(device)

# Loading the learnt model 
actor.load_state_dict(torch.load('/home/aman/Programming/RL-Project/Deterministic-GAIL/weights/GAIL_actor.pth'))
actor.eval()

# Defining tensorboard agent
writer = SummaryWriter(logs_dir+"/Testing-{}".format(exp_num))

def write_arr_tb(arr,writer,name,start_index):
   """
    arr         : 1D array to write to tensorboard
    writer      : tensorboard summary writer object
    name        : name to specify on tensorboard
    start_index : starting index for the array to write
   """
   for i in range(arr.shape[0]):
       writer.add_scalar(name,arr[i],start_index+i) 


if random_seed:
    # Setting seed value for numpy and torch
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)    
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# make necessary directories
os.makedirs(logs_dir,exist_ok=True)    

start = 0

eval_returns = np.zeros(test_episodes,dtype=np.float)

for eps in range(test_episodes):

    if np.mod(eps, 3) == 0:
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    rewards = []
    for steps in range(max_steps):
        
        # state in GPU
        state = torch.tensor(state,dtype=torch.float,device=device).view(1,-1)
        
        # get action from policy trained in last epoch    
        action = actor(state).cpu().data.numpy().flatten()
        
        # take action and observe reward and next state     
        ob, reward, done, info = env.step(action)
        next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        rewards.append(reward)

        if done:
            break

        state = next_state

    _return =  calc_disc_return(rewards,gamma)
    eval_returns[eps] = _return[0]
    writer.add_scalar("Test/mean_return",_return[0],eps)
