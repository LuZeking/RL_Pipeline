from turtle import forward
import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module

class QNet(Module):
    def __init__(self, dims:[int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = built_mlp([state_dim, *dims, action_dim])
        self.explore_rate = None
        self.action_dim = action_dim

    def forward(state: Tensor) -> Tensor:
        
        return self.net[state]  #Q values for multi actions, equal to argmax in 

    def get_action(self, state: Tensor) -> Tensor:
        if self.explore_rate < torch.rand(1):
            action = self.net(state).argmax(dim=1,keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0],1))
        return action

class Actor(Module):
    """policy gradient actor"""
    def forward():
        return action.tanh()
    pass

class Critic(Module):
    """policy gradient critic"""
    pass

class ActorPPO():
    def __init__(self, dims:[int], state_dim: int, action_dim : int) -> None:
        super.__init__()
        self.net = built_mlp(dims = [state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zero((1, action_dim)), requires_grad= True) # trainable parameter
    
    def forward(self, state: Tensor) ->Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        
        """#! Normal distribution , log prob ??? sum()?? action?
        """
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):

        """#! Normal distribution , log prob ??? sum()?? entropy 
        """
        return logprob, entropy
    
    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(Module):
    def __init__(self, dims:[int], state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = built_mlp([state_dim,*dims, 1])  # value_dim =1 
    
    def forward(self, state: Tensor) -> Tensor:
        return self.net(state) # advantage value

def built_mlp(dims: [int]) -> nn.Sequential: # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims)-1)：
        net_list.extend([nn.Linear(dims[i], dims[i+1]), nn.RELU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)
