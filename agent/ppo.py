import numpy as np
import torch
import torch.nn.functional as F
import random
from gcn import GNNActor, GNNCritic
import os
from observation import BipartiteNodeObs
    
class PPO:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.gamma = args.gamma
        self.critic_lr = 5e-4
        self.actor_lr = 1e-4

        self.lmbda = 0.95
        self.epoch = 10               
        self.eps = 0.2          

        self.entropy_coef = args.entropy_coef


        self.actor_net = GNNActor(args).to(self.device)
        self.eval_critic_net = GNNCritic(args).to(self.device)


        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.eval_critic_net.parameters(), lr=self.critic_lr, weight_decay=1e-5)

        self.tau = 0.01

        self.train_cnt = 0
        self.save_cycle = 1000

        self.epsilon = 10
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.5


    def to_cpu(self):
        self.device = torch.device('cpu')
        self.actor_net.to(self.device)
        self.eval_critic_net.to(self.device)
        

    def to_gpu(self,args):
        self.device = args.device
        self.actor_net.to(self.device)
        self.eval_critic_net.to(self.device)

    def choose_action(self, state: BipartiteNodeObs, action_set):
        constraint_features = state.constraint_features
        edge_index = state.edge_indices
        edge_attr = state.edge_features
        variable_features = state.variable_features

        constraint_features = torch.tensor(constraint_features, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
        variable_features = torch.tensor(variable_features, dtype=torch.float32).to(self.device)

       
        policy_logits = self.actor_net(
            constraint_features, edge_index, edge_attr, variable_features
        )
        for i in range(policy_logits.shape[0]):
            if i not in action_set:
                policy_logits[i] = -1e8

        action_dis = F.softmax(policy_logits, dim=-1)
        action = int(torch.multinomial(action_dis,1).cpu())

        log_action_prob = torch.log(
            action_dis[action]
        ).cpu().detach().numpy()
        
        return action, log_action_prob

    def evaluate(self, state: BipartiteNodeObs, action_set):
        constraint_features = state.constraint_features
        edge_index = state.edge_indices
        edge_attr = state.edge_features
        variable_features = state.variable_features

        constraint_features = torch.tensor(constraint_features, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
        variable_features = torch.tensor(variable_features, dtype=torch.float32).to(self.device)

       
        policy_logits = self.actor_net(
            constraint_features, edge_index, edge_attr, variable_features
        )
        for i in range(policy_logits.shape[0]):
            if i not in action_set:
                policy_logits[i] = -1e8
        action_dis = F.softmax(policy_logits, dim=-1)
        action = int(torch.argmax(action_dis).cpu())

        log_action_prob = torch.log(
            action_dis[action]
        ).cpu().detach().numpy()
        
        return action, log_action_prob
    
    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def soft_update(self, eval_net, target_net):
        """soft update"""
        for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_(self.tau*eval_param.data + (1.0-self.tau)*target_param.data)
    
    def get_policy_probs(self, constraint_features, edge_index, edge_attr, variable_features, batch_size, action_set):
        policy_logits_ori = self.actor_net(
            constraint_features, edge_index, edge_attr, variable_features
        )
        policy_logits = policy_logits_ori.view(batch_size,-1)

        policy_probs_list = []
        for i in range(batch_size):
            one_policy_logits = policy_logits[i]
            one_policy_dis = F.softmax(one_policy_logits[action_set[i]])
            one_policy_probs = torch.zeros(policy_logits.shape[1]).to(self.device)
            one_policy_probs[action_set[i]] = one_policy_dis
            policy_probs_list.append(one_policy_probs)


        policy_probs = torch.vstack(policy_probs_list)
        return policy_probs

    def train(self,batch):


        constraint_features = batch['constraint_features']
        edge_index = batch['edge_index']
        edge_attr = batch['edge_attr']
        variable_features = batch['variable_features']
        action = batch['action']
        reward = batch['reward']
        action_set = batch['action_set']
        action_set_size = batch['action_set_len']

        next_constraint_features = batch['next_constraint_features']
        next_edge_index = batch['next_edge_index']
        next_edge_attr = batch['next_edge_attr']
        next_variable_features = batch['next_variable_features']
        done = batch['done']

        next_action_set = batch['next_action_set']
        next_action_set_size = batch['next_action_set_len']
        
        action_set_ori = batch['action_set_ori']

        constraint_features = torch.tensor(constraint_features, dtype=torch.float32).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(self.device)
        variable_features = torch.tensor(variable_features, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action_set = torch.tensor(action_set, dtype=torch.long).to(self.device)
        action_set_size = torch.tensor(action_set_size, dtype=torch.long).to(self.device)

        next_constraint_features = torch.tensor(next_constraint_features, dtype=torch.float32).to(self.device)
        next_edge_index = torch.tensor(next_edge_index, dtype=torch.long).to(self.device)
        next_edge_attr = torch.tensor(next_edge_attr, dtype=torch.float32).to(self.device)
        next_variable_features = torch.tensor(next_variable_features, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.long).to(self.device)
        
        next_action_set = torch.tensor(next_action_set, dtype=torch.long).to(self.device)
        next_action_set_size = torch.tensor(next_action_set_size, dtype=torch.long).to(self.device)
        
        batch_size = reward.shape[0]

       
        td_target = reward + self.gamma * self.eval_critic_net(
            next_constraint_features, next_edge_index, next_edge_attr, next_variable_features, batch_size
        ).view(-1, 1) * (1 - done)

        td_delta = td_target - self.eval_critic_net(
            constraint_features, edge_index, edge_attr, variable_features, batch_size
        ).view(-1, 1)

        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        
        policy_probs = self.get_policy_probs(
            constraint_features, edge_index, edge_attr, variable_features, batch_size, action_set_ori
        )
        old_log_probs = torch.log(
            policy_probs.gather(1, action).detach() + 1e-20
        )

        for _ in range(self.epoch):
            policy_probs = self.get_policy_probs(
                constraint_features, edge_index, edge_attr, variable_features, batch_size, action_set_ori
            )
            entropy = -torch.sum(
                (policy_probs+1e-20) * torch.log(policy_probs+1e-20), dim=-1
            ).view(-1,1)

            log_probs = torch.log(policy_probs.gather(1, action) + 1e-20)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            actor_loss = torch.mean(
                -torch.min(surr1, surr2) - self.entropy_coef * entropy
            ) 
            critic_loss = torch.mean(
                F.mse_loss(self.eval_critic_net(
                    constraint_features, edge_index, edge_attr, variable_features, batch_size
                ).view(-1, 1), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        if self.train_cnt % self.save_cycle == 0:
            self.save_model()

        self.train_cnt += 1

        self.epsilon = self.epsilon * self.epsilon_decay \
            if self.epsilon >= self.min_epsilon else self.min_epsilon


    def save_current_model(self):
        folder_name = "model/{}".format(self.args.run_id)
        path = "./" + folder_name
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor_net.state_dict(),
            "./model/{}/{}.pkl".format(self.args.run_id, "ppo")
        )

    def save_model(self):
        folder_name = "model/{}".format(self.args.run_id)
        path = "./" + folder_name
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor_net.state_dict(),
            "./model/{}/{}.pkl".format(self.args.run_id, self.train_cnt)
        )
        