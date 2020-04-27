from gym_torcs import TorcsEnv
import torch
import numpy as np
from gail import GAIL
from torch.utils.tensorboard import SummaryWriter
from utility import calc_disc_return
import os

#######################################
#                                     #  
#          Hyperparameters            #  
#                                     #   
#######################################

lr = 0.0001                           # learning rate
random_seed = 100                     # random seed for experiment  
betas = (0.5,0.99)                    # betas for adam optimizer
epochs = 1000                         # number of epochs to train
n_iter = 512                          # number of iterations for each update   
mini_batch = 1024                     # number of transitions sampled from expert
gamma = 0.95                          # discount factor
eval_episodes = 2                     # number of evaluation episodes for validation reward
VISION = False                        # using torcs in measurement based mode (instead vision based mode)  
max_steps = 50000                     # maximum time steps for each episode  
expert_thresh = 0.5                   # parameter to control number of exper trajectories selected (lower means more expert traj.)  


expert_dir  = "/home/aman/Programming/RL-Project/expert_trajectories" # directory containing expert trajectories
weights_dir = "./weights"             # directory to store weights
logs_dir    = "./logs"                # directory to store tensorboard logs  


# Defining torcs environment
env = TorcsEnv(vision=VISION,throttle=True,gear_change=False)
state_size = 29                       # number of parameters in state space
action_size = 3                       # number of parameters in action space



save_freq  = 5                        # number of epochs after which to save model      
exp_num = 1                           # experiment number

# Define the GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected {} device".format(device))


# Defining the GAIL agent
agent = GAIL(expert_dir,expert_thresh,state_size,action_size,lr,betas,device,gamma)

# defining tensorboard agent
writer = SummaryWriter(logs_dir+"/Experiment-{}".format(exp_num))

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
os.makedirs(weights_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)    

start = 0
for epoch in range(1,epochs+1):
     
    # update the policy
    actor_loss,disc_loss = agent.update(n_iter,batch_size=mini_batch)
    
    # add the losses to tensorboard
    write_arr_tb(actor_loss,writer,"Training/actor_loss",start)
    write_arr_tb(disc_loss,writer,"Training/discriminator_loss",start)

    start += actor_loss.shape[0]

    '''
    Evaluate the policy learnt in last network
    '''
    eval_returns = np.zeros(eval_episodes,dtype=np.float)
    for eps in range(eval_episodes):
    
        if np.mod(eps, 3) == 0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
        rewards = []
        for steps in range(max_steps):
            
            # get action from policy trained in last epoch    
            action = agent.get_action(state)
            
            # take action and observe reward and next state     
            ob, reward, done, info = env.step(action)
            next_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            rewards.append(reward)

            if done:
                break

            state = next_state

        _return =  calc_disc_return(rewards,gamma)
        eval_returns[eps] = _return[0]

    mean_return = np.mean(eval_returns)
    writer.add_scalar("Eval/mean_return",mean_return,epoch)

    if epoch % save_freq == 0:
        agent.save(name='GAIL-{}'.format(random_seed))
        





     

