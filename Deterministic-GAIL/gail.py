from actor import ActorNetwork
from discriminator import Discriminator
from getexptraj import ExpertTrajectories
import torch
import numpy as np

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

class GAIL:

    def __init__(self, exp_dir, exp_thresh, state_dim, action_dim, learn_rate, betas, _device,_gamma,load_weights=False):
        """
            exp_dir : directory containing the expert episodes
         exp_thresh : parameter to control number of episodes to load 
                      as expert based on returns (lower means more episodes)
          state_dim : dimesnion of state 
         action_dim : dimesnion of action
         learn_rate : learning rate for optimizer 
            _device : GPU or cpu
            _gamma  : discount factor
     _load_weights  : load weights from directory
        """

        # storing runtime device
        self.device = _device

        # discount factor
        self.gamma = _gamma

        # Expert trajectory
        self.expert  = ExpertTrajectories(exp_dir,exp_thresh,gamma=self.gamma)  
        
        # Defining the actor and its optimizer
        self.actor       = ActorNetwork(state_dim).to(self.device)
        self.optim_actor = torch.optim.Adam(self.actor.parameters(),lr=learn_rate,betas=betas) 

        # Defining the discriminator and its optimizer
        self.disc      = Discriminator(state_dim,action_dim).to(self.device)
        self.optim_disc = torch.optim.Adam(self.disc.parameters(),lr=learn_rate,betas=betas) 

        if not load_weights:
            self.actor.apply(init_weights)
            self.disc.apply(init_weights)
        else:
            self.load()

        # Loss function crtiterion
        self.criterion = torch.nn.BCELoss()

    def get_action(self,state):
        """
            obtain action for a given state using actor network 
        """
        state = torch.tensor(state,dtype=torch.float,device=self.device).view(1,-1)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self,n_iter,batch_size=100):
        """
            train discriminator and actor for mini-batch
        """ 
        # memory to store 
        disc_losses = np.zeros(n_iter,dtype=np.float)
        act_losses  = np.zeros(n_iter,dtype=np.float)

        
        for i in range(n_iter):

            # Get expert state and actions batch
            exp_states,exp_actions = self.expert.sample(batch_size)
            exp_states  = torch.FloatTensor(exp_states).to(self.device)
            exp_actions = torch.FloatTensor(exp_actions).to(self.device)
            
            # Get state, and actions using actor
            states,_ = self.expert.sample(batch_size)
            states  = torch.FloatTensor(states).to(self.device)
            actions = self.actor(states)

            '''
                train the discriminator
            '''
            self.optim_disc.zero_grad()
            
            # label tensors    
            exp_labels    = torch.full((batch_size,1),1,device=self.device)
            policy_labels = torch.full((batch_size,1),0,device=self.device)

            # with expert transitions
            prob_exp = self.disc(exp_states,exp_actions)
            exp_loss = self.criterion(prob_exp,exp_labels)

            # with policy actor transitions
            prob_policy = self.disc(states,actions.detach())
            policy_loss = self.criterion(prob_policy,policy_labels)

            # use backprop
            disc_loss = exp_loss + policy_loss
            disc_losses[i] = disc_loss.mean().item()

            disc_loss.backward()
            self.optim_disc.step()

            '''
                train the actor
            '''
            self.optim_actor.zero_grad()
            loss_actor = -self.disc(states,actions)
            act_losses[i] = loss_actor.mean().detach().item()

            loss_actor.mean().backward()
            self.optim_actor.step()

        print("Finished training minibatch")

        return act_losses,disc_losses

    def save(self, directory='./weights', name='GAIL'):
        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(directory,name))
        torch.save(self.disc.state_dict(), '{}/{}_discriminator.pth'.format(directory,name))
        
    def load(self, directory='./weights', name='GAIL'):
        self.actor.load_state_dict(torch.load('{}/{}_actor.pth'.format(directory,name)))
        self.disc.load_state_dict(torch.load('{}/{}_discriminator.pth'.format(directory,name)))

    def set_mode(self,mode="train"):
        
        if mode == "train":
            self.actor.train()
            self.disc.train()
        else:
            self.actor.eval()
            self.disc.eval()









