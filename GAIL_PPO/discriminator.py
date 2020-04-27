import torch
import torch.nn as nn

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class Discriminator(nn.Module):
    
    def __init__(self,state_dim,action_dim):
        super(Discriminator,self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim,HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS,HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS,1)
    
    def forward(self,state,action):
        state_action = torch.cat([state,action],1)
        x = torch.tanh(self.fc1(state_action))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
        
