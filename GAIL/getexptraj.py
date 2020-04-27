import pickle
import glob
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from scipy import signal

class ExpertTrajectories(object):

    def __init__(self,_expert_dir,thresh = 0.2,_num_exp_traj=None,gamma=0.95):
        self.expert_dir = _expert_dir
        self.gamma = gamma
        self.use_thresh = thresh is not None

        '''
        Use threshold
        '''
        if self.use_thresh:
            assert thresh > 0 and thresh <= 1 , "threshold value should be between 0 and 1"
            self.thresh = thresh
        
        '''
        Use fixed number of trajectories
        '''            
        self.use_fixed_exp_traj = _num_exp_traj is not None
        if self.use_fixed_exp_traj:
            assert _num_exp_traj > 0 , "Number of expert trajectories should be greater than 0"
            self.num_exp_traj = _num_exp_traj

        assert self.use_fixed_exp_traj ^ self.use_thresh , "Use one of the two methods : \n (1) threshold for number of trajectories(lower means more trajectories)\n (2) Fixed number of expert trajectories"

        self.chos_states,self.chos_actions = self.__load_state_action()

    def __get_return(self,episode):
        """
        Calculate non-discounted return for the episode
        """
        return np.sum(episode['reward'])
    
    def __read_episode(self,episodes,file,index):
        """
        Read episode PKL file from 
            file : PKL file path
        episodes : memory to store the data read from file 
                   (used as global memory for multi-threaded pool read)
           index : index in the memory to store the data    
        """
        episodes[index] = pickle.load(open(file,"rb"))        
   
    def __get_disc_rewards(self,episode):
        """
            Calculate discounted return for the episode
        """
        rewards = np.array(episode['reward'])
        disc_return = signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        return disc_return

    def __get_eps_length(self,episode):
        """
            Calculate length for the episode
        """
        states = np.array(episode['state'])
        return states.shape[0] 


    def __get_expert_episodes(self):
        """
        Read the entire episode data from which qualifies the thresholding method
        """
        
        # loading the expert episodic data
        files = np.asarray(glob.glob(os.path.join(self.expert_dir,"*.pkl")))
        
        episodes = [None for file in files]
        with ThreadPoolExecutor(max_workers=40) as executor:
            for i,file in enumerate(files):
                executor.submit(self.__read_episode,episodes,file,i)
        # episodes = np.asarray(episodes)
        
        # Calculate returns
        returns = np.array([self.__get_disc_rewards(episode)[0] for episode in episodes])

        # Calculate mean return
        mean_return = np.mean(returns)

        # Calculate episode length
        length_eps = np.array([self.__get_eps_length(episode) for episode in episodes]) 

        # Calculate max return
        max_len = np.max(length_eps)

        if self.use_thresh:
            # Obtain all the episodes for which rerturn is greater than mean return and length is greater than mean length
            indxs = np.where((returns > self.thresh * mean_return) 
                                & (length_eps > self.thresh * max_len))[0]
            episodes =  [episodes[i] for i in indxs]

        if self.use_fixed_exp_traj:
            assert self.num_exp_traj <= episodes.shape[0], "Number of expert trajectories expected greater than total number of trajectories = {}".format(episodes.shape[0])
            episodes = episodes[-1*self.num_exp_traj:]

        return episodes


    def __load_state_action(self):
        """
           helper method to read state and actions of all the data in selected episodes  
        """
        episodes = self.__get_expert_episodes()
        states = []
        actions = []
        for i in range(len(episodes)): 
            states  = states  + episodes[i]['state']
            actions = actions + episodes[i]['action']
        
        states  = np.array(states)
        actions = np.array(actions)

        return states,actions

    def sample(self,batch_size):
        """
            sample a random batch size of states and actions
        """
        assert self.chos_states.shape[0] >= batch_size, "batch size greater than total experience size"        
        batch_indxs = np.random.choice(np.arange(self.chos_states.shape[0]),size=batch_size)

        return self.chos_states[batch_indxs],self.chos_actions[batch_indxs]


if __name__ == "__main__":
    exp_dir = "./expert_trajectories"
    exp_traj = ExpertTrajectories(_expert_dir=exp_dir,thresh=0.5)
    states,actions = exp_traj.sample(100)
    print(states)
