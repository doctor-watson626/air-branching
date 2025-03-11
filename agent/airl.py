# from discriminators.base import Discriminator
# from networks.discriminator_network import G,H
import torch
import torch.nn as nn
from gcn import GNNActor, GNNCritic, GNNDis
import torch_geometric
import pickle
import gzip
import os
import numpy as np
from pathlib import Path
from replay_buffer import PPOReplayBuffer
from agent.ppo import PPO
from utils import pad_tensor
import torch.nn.functional as F
import random
class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        candidates,
        nb_candidates,
        candidate_choice,
        candidate_scores,

        action_ori,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores
        self.action_ori = action_ori
       

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):

        with gzip.open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample

        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)

        try:
            variable_features = sample_observation.column_features
        except:
            variable_features = sample_observation.variable_features


        candidates = np.array(sample_action_set, dtype=np.int32)
        candidate_scores = np.array([sample_scores[j] for j in candidates])
        candidate_choice = np.where(candidates == sample_action)[0][0]



        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(candidates),
            len(candidates),
            torch.LongTensor([candidate_choice]),
            torch.FloatTensor(candidate_scores),

            torch.LongTensor(sample_action),

        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph

class AIRL(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args.device
        self.f = GNNDis(args)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
    def get_f(self,state, action_index, action_set, action_set_len):
        logits = self.f(
            state[0], state[1], state[2], state[3],
        )

        logits = pad_tensor(logits[action_set], action_set_len)
        logits = logits.gather(1, action_index)
        return logits

    
    def get_d(self,log_prob,state,action_index, action_set, action_set_len):
        exp_f = torch.exp(self.get_f(state, action_index, action_set, action_set_len))
        return (exp_f/(exp_f + torch.exp(log_prob)))
    
    def get_reward(self,log_prob, state, action_index, action_set, action_set_len):
        d = (self.get_d(log_prob, state, action_index, action_set, action_set_len)).detach()
        return (torch.log(d + 1e-3) - torch.log((1-d)+1e-3))
        
    def forward(self,log_prob,state, action_index, action_set, action_set_len):
        d = (self.get_d(log_prob,state, action_index, action_set, action_set_len))
        return d

    def train_network(self,
        agent_s, agent_a_index, agent_log_prob, agent_action_set, agent_action_set_len,
        strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len, is_pretrain=False,
        pseudo_s = None, pseudo_a_index = None, pseudo_log_prob = None, pseudo_action_set = None, pseudo_action_set_len = None,
    ):

        strong_preds = self.forward(strong_log_prob, strong_s,strong_a_index, strong_action_set, strong_action_set_len)
        strong_loss = self.criterion(strong_preds,torch.ones(strong_preds.shape[0],1).to(self.device)) 

        agent_preds = self.forward(agent_log_prob, agent_s, agent_a_index, agent_action_set, agent_action_set_len)
        agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device)) 
        
        if self.args.is_three_loss and not is_pretrain:
            pseudo_preds = self.forward(pseudo_log_prob, pseudo_s, pseudo_a_index, pseudo_action_set, pseudo_action_set_len)
            pseudo_loss = self.criterion(pseudo_preds,torch.zeros(pseudo_preds.shape[0],1).to(self.device)) 


        loss = strong_loss + agent_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def batch_to_input(strong_batch, agent: PPO):

    strong_s = [
            strong_batch.constraint_features,
            strong_batch.edge_index,
            strong_batch.edge_attr,
            strong_batch.variable_features
        ]
    strong_a = strong_batch.candidate_choices.view(-1,1)
    policy_logits_ori = agent.actor_net(
        strong_batch.constraint_features, strong_batch.edge_index, 
        strong_batch.edge_attr, strong_batch.variable_features,
    )

    policy_logits = pad_tensor(
        policy_logits_ori[strong_batch.candidates], strong_batch.nb_candidates
    )

    strong_probs = F.softmax(policy_logits, dim=-1)
    strong_log_prob = torch.log(
        strong_probs.gather(1, strong_a) + 1e-20
    )



    return strong_s, strong_a, strong_log_prob, strong_batch.candidates, strong_batch.nb_candidates

def batch_to_input_pretrain(strong_batch, ):
    strong_s = [
            strong_batch.constraint_features,
            strong_batch.edge_index,
            strong_batch.edge_attr,
            strong_batch.variable_features
        ]
    strong_a = strong_batch.candidate_choices.view(-1,1)

    candidate_scores = pad_tensor(
        strong_batch.candidate_scores, strong_batch.nb_candidates, 1e6
    )
    
    bad_action_probs = F.softmax(-candidate_scores, dim=-1)
    strong_log_prob = torch.log(
        bad_action_probs.gather(1, strong_a) + 1e-20
    )

    return strong_s, strong_a, strong_log_prob, strong_batch.candidates, strong_batch.nb_candidates

def get_dataloader(args):
    strong_sample_files = [
        str(path) for path in Path("samples/{}/".format(args.instance_name)).glob("sample_*.pkl")
    ]
    pseudo_sample_files = [
        str(path) for path in Path("samples/{}/".format(args.instance_name)).glob("bad_sample_*.pkl")
    ]
    strong_dataset = GraphDataset(strong_sample_files)
    pseudo_dataset = GraphDataset(pseudo_sample_files)
    strong_loader = torch_geometric.data.DataLoader(strong_dataset, batch_size=args.batch_size, shuffle=True)
    pseudo_loader = torch_geometric.data.DataLoader(pseudo_dataset, batch_size=args.batch_size, shuffle=True)
    return strong_loader, pseudo_loader

def train_airl(discriminator: AIRL, D_env: PPOReplayBuffer, agent: PPO, strong_loader, pseudo_loader, args):
    

    for _ in range(args.airl_train_epoch):
    # for i, (strong_batch, pseudo_batch) in enumerate(zip(strong_loader, pseudo_loader)):
        strong_batch = next(strong_loader)
        pseudo_batch = next(pseudo_loader)
        agent_batch = D_env.sample_one_batch(args.batch_size)
        strong_batch = strong_batch.to(args.device)
        pseudo_batch = pseudo_batch.to(args.device)
        
        strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len = batch_to_input(strong_batch, agent)
        if args.is_three_loss:
            pseudo_s, pseudo_a_index, pseudo_log_prob, pseudo_action_set, pseudo_action_set_len = batch_to_input(pseudo_batch, agent)

        agent_batch_size = agent_batch["reward"].shape[0]

        constraint_features = torch.tensor(agent_batch["constraint_features"], dtype=torch.float32).to(args.device)
        edge_index = torch.tensor(agent_batch["edge_index"], dtype=torch.long).to(args.device)
        edge_attr = torch.tensor(agent_batch["edge_attr"], dtype=torch.float32).to(args.device)
        variable_features = torch.tensor(agent_batch["variable_features"], dtype=torch.float32).to(args.device)
        
        agent_s = [constraint_features, edge_index, edge_attr, variable_features]
            
        agent_a = agent_batch["action"]
        action_set_ori = agent_batch["action_set_ori"]
        action_index_list = []
        for batch_idx, ori_set in enumerate(action_set_ori):
            action_index_list.append(
                np.where(ori_set == agent_a[batch_idx,0])[0][0]
            )

        agent_a = torch.tensor(agent_a, dtype=torch.long).to(args.device).view(-1,1)
        agent_a_index = torch.tensor(action_index_list, dtype=torch.long).to(args.device).view(-1,1)


        action_set_len = torch.tensor(agent_batch["action_set_len"], dtype=torch.long).to(args.device)
        action_set = torch.tensor(agent_batch["action_set"], dtype=torch.long).to(args.device)

        policy_probs = agent.get_policy_probs(
            constraint_features, edge_index, edge_attr, variable_features,
            agent_batch_size, agent_batch["action_set_ori"]
        )
        agent_log_prob = torch.log(
            policy_probs.gather(1, agent_a) + 1e-20
        )

        
        if args.is_three_loss:
            loss = discriminator.train_network(
                agent_s, agent_a_index, agent_log_prob, action_set, action_set_len,
                strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len, False,
                pseudo_s, pseudo_a_index, pseudo_log_prob, pseudo_action_set, pseudo_action_set_len,
            )
        else:
            loss = discriminator.train_network(
                agent_s, agent_a_index, agent_log_prob, action_set, action_set_len,
                strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len,
            )
    return loss

def pretrain_airl(discriminator: AIRL, args):
    

    strong_loader, pseudo_loader = get_dataloader(args)
    for epoch in range(args.airl_pretrain_epoch):
        for i, (strong_batch, pseudo_batch) in enumerate(zip(strong_loader, pseudo_loader)):

            strong_batch = strong_batch.to(args.device)
            pseudo_batch = pseudo_batch.to(args.device)
            
            strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len = batch_to_input_pretrain(strong_batch)
            pseudo_s, pseudo_a_index, pseudo_log_prob, pseudo_action_set, pseudo_action_set_len = batch_to_input_pretrain(pseudo_batch)
            
            loss = discriminator.train_network(
                pseudo_s, pseudo_a_index, pseudo_log_prob, pseudo_action_set, pseudo_action_set_len,
                strong_s, strong_a_index, strong_log_prob, strong_action_set, strong_action_set_len, True,
                None, None, None, None, None,
            )
            
        
        print("pretrain epoch: ", epoch)
    
    return loss