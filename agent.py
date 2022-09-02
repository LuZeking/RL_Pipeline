from curses.ascii import BS
import torch
from copy import deepcopy
from torch import Tensor
from net import QNet  # DQN
from net import Actor, Critic  # DDPG
from net import ActorPPO, CriticPPO  # PPO
from config import Config
import torch

class AgentBase(torch.nn.Module):

    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id =0, args: Config = Config()) -> None:
        super().__init__()
        """"""
        self.xxx_params =
        self.gamma =
        self.state_dim = 
        self.action_dim =
        self.batch_size =
        self.reapt_times = 
        self.soft_update_tau = 

        self.states = None  # assert self.state == (1, state_dim)
        self.device = 

        self.learning_rate = 

        self.reward_scale = 



        act_class = getatrr(self, act_class,  None)
        cri_class = getatrr(self, cri_class, None)
        self.act = self.act_target= act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = act_class(net_dims, state_dim, action_dim).to(self.device)\
            if cri_class else self.act 

        self.act_optimizer = 
        self.cri_optmizer = 

        self.criterion = torch.nn.SmoothL1Loss()
        

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        """copy data form current & target net params"""
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data*tau + tar.data*(1.0-tau))
        

class AgentDQN(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id=0, args: Config = Config()) -> None:
        self.act_class = getattr(self,"act_class", QNet)
        self.cri_class = getattr(self,"cri_class", None)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.act_target = deepcopy(self.act_class)
        self.cri_target = deepcopy(self.cri_class)

        self.act.explore_rate = getattr(args, "explore_rate", 0.25)  # Get paras value from args, using getattr when it is not exist

    def explore_env(self, env, horizon_len, if_random) -> [Tensor]:  # ->  buffer_items
        states = torch.zeros((horizon_len, self.state_dim), dtype= torch.float32).to(self.device)
        actions =  torch.zeros((horizon_len, 1), dtype= torch.int).to(self.device)
        rewards = torch.ones(horizon_len, dtype= torch.float32).to(self.device)
        dones =  torch.zeros(horizon_len, dtype= torch.bool).to(self.device)

        ary_state = self.states[0]
        get_action = self.act.get_action
        for i in range(horizon_len): #*？ horizon_len != batch_size ??
            state = torch.as_tensor(ary_state, dtype = torch.float32, device =self.device)
            action = torch.randint(self.action_dim, size = (1,))[0] if if_random else get_action(state.unsqueeze(0))[0,0] #! ? why [0,0]

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()
            
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            dones[i] = done

        self.state[0] = ary_state  #! ? why self.state[0] = state
        rewards = rewards.unsqueeze(1)
        undones = (1.0-dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]: # training_logging
        '''update Qnet'''
        #s,a = 
        obj_critics = q_values =0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        #* fixed cri_target for a while, which means updating in a certain batch steps
        for i in range(update_times):
            obj_critic, q_value = self.get_obj_critic(buffer, self.batch_size) #*  get critic from (q_value, q_label) to update
            self.optimizer_update(self.cri_optmizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item() #* accumulate the criterion & Q value, then divided by n to get mean value
            q_values += q_value.item()
        return obj_critic / update_times, q_values / update_times 

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample()
            next_q = self.cri_target(next_state).max(dim=1, keepdim=True)[0] #! why max dim =1 [0]
            q_label = reward + undone* self.gamma * next_q

        q_value = self.cri(state).gather(1, action.long()) #! why gather
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value.mean()

class DDPGAgent(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', Actor)  # get the attribute of object `self`, set Actor in default
        self.cri_class = getattr(self, 'cri_class', Critic)  # get the attribute of object `self`, set Critic in default
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.act_target = deepcopy(self.act)
        self.cri_target = deepcopy(self.cri)

        self.act.explore_noise_std = getattr(args, "explore_noise", 0.1)

    def explore_env(env, horizon_len, if_random) -> buffer_items:
        pass
    
    def update_net(self, buffer) -> training_logging:
        '''update Actor & Critic, here Q value is the target of Actor, Q instead of V in DQN
           The difference can be found in #* 1 2 3 4 5 6, the key is #* 2&3， which makes critic action as the actor' target
        '''
        obj_critics = q_values =0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size) #*  1
            self.optimizer_update(self.cri_optmizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item() #* accumulate the criterion & Q value, then divided by n to get mean value
            
            action = self.act(state) #* 2 
            obj_actor = self.cri_target(state, action).mean() #* 3 
            self.optimizer_update(self.act_optimizer, -obj_actor) #*
            self.soft_update_tau(self.act_target, self.act, self.soft_update_tau) #*
            obj_actors += obj_actor.item() #* 6 for DQN, it used to be q_values += q_value.item()

        return obj_critic / update_times, q_values / update_times 
    
    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        """Similar to DQN' get_obj_critic , DDPG = AC+DQN -> determinstric， here Critic is Q not V 
           The difference can be found in #* 1 2 3
        """
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample()
            next_action = self.act_target(next_state) #* 1
            next_q = self.cri_target(next_state, next_action,) #* 2 in DQN just self.cri_target(next_q).max(dim=1, keepdim=True)[0] #! why max dim =1 [0]
            q_label = reward + undone* self.gamma * next_q

        q_value = self.cri(state, action) #* 3
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value.mean()

class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id=0, args: Config = Config()) -> None:
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratop_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95) # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(env, horizon_len, if_random) -> buffer_items:
        pass
    
    def update_net(self,buffer) -> training_logging:
        '''update PPO Actor & Critic, here Q value is the measurement of Actor'''
        
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            BATCHSIZE = 2**10
            vlaues = [self.cri(states[1:1+BATCHSIZE]) for i in range(0, buffer_size, BATCHSIZE)]
            values = torch.cat(values, dim =0).squeeze(1)

            advantages = self.get_advantages(rewards, undones, values)
            reward_sums = advantages + values
            del rewards, undones, vlaues # ! ?why del these?

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0)+1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)
                
        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size = (self.batch_size,), requires_grad= False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value,reward_sum)  #* criterion measurement for Critics
            self.optimizer_update(self.cri_optmizer, obj_critic)

            new_log_prob, obj_entropy = self.act.get_logprob_entropy(state,action)
            ratio = (new_log_prob - logprob.detach()).exp()  #! ? why new_log_prob- log_prob
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor =  obj_surrogate - obj_entropy.mean()* self.lambda_entropy  #* criterion measurement for actor
            self.optimizer_update(self.act_optimizer, -obj_actor) #! why - obj_actor

            obj_critics += obj_critic.item()
            obj_actors += obj_actor


        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values) # advantage_value

        masks = undones*self.gamma
        horizon_len = rewards.shape[0]  #? why == reward.shape[0]

        next_state = torch.Tensor(self.states, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state).detach()[0,0] #! why always [0,0]

        advantage = 0 # last gae_lambda
        for t in range(horizon_len-1, -1, -1):
            delta = rewards[t] + masks[t]*next_value-values[t]



class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0 # pointer
        self.is_full = False
        self.cur_size = 0
        self.max_size = max_size

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_availble() and (gpu_id >= 0)) else "cpu" )

        self.states = torch.empty((max_size,state_dim),dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)    

    def update(self, items:[Tensor]):
        states, actions, rewards, undones = items
        p = self.p +rewards.shape[0]  # pointer move to items' end to choose range
        if p > self.max_size:
            self.if_full = True

            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p -self.max_size

            self.states[p0:p1] = states[:p2]  # store in snake cirle, one part stored in the end, another stored at the begaining
            self.states[0:p] = states[-p:]

            self.actions[p0:p1] = actions[:p2]
            self.actions[0:p] = actions[-p:]

            self.rewards[p0:p1] = rewards[:p2]
            self.rewards[0:p] = rewards[-p:]

            self.undones[p0:p1] = undones[:p2]
            self.undones[0:p] = undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p # move
        self.cur_size = self.max_size if self.if_full else self.p

    
    def sample(self, batch_size: int) -> [Tensor]:
        indices = torch.randint(self.cur_size-1, size=(batch_size,), requires_grad= False)  # sample randomly batchsize in range(0,cur_size-1)
        return (self.states[indices],  
                self.actions[indices], 
                self.rewards[indices], 
                self.undones[indices],
                self.states[indices],  # next state
                )
