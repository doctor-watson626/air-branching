import numpy as np
import scipy.sparse as sp
import torch 
import torch.nn.functional as F
from observation import BipartiteNodeObs
import ecole
from env import RealEnv
from reward import BinarySolved, PrimalDualGapFrac, BinaryFathomed
from scipy.stats import gmean
import random

def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output



def evaluate_agent(agent,args,fixed_instances,evaluate_num=5, discriminator = None):
    evaluate_limit_time =  120
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": evaluate_limit_time,
    }
    
    ecole_env = ecole.environment.Branching(
        observation_function = ecole.observation.NodeBipartite(),
        scip_params=scip_parameters,

        reward_function = BinaryFathomed()
    )
    

    real_env = RealEnv(ecole_env)

    reward_list = []
    nnodes_list = []
    for i in range(evaluate_num):
        
        done = True
        while done:
            scip_model = next(fixed_instances)
            observation, action_set, _, done, info = real_env.reset(scip_model)

        total_reward = 0
        while not done:
            action, log_action_prob = agent.evaluate(observation, action_set)
            
            if discriminator is not None:
                action_index = np.where(action_set == action)[0][0]
                dis_reward = discriminator.get_reward(
                    torch.tensor(log_action_prob, dtype=torch.float32).view(1,1).to(args.device),
                    [
                        torch.tensor(observation.constraint_features, dtype=torch.float32).to(args.device),
                        torch.tensor(observation.edge_indices, dtype=torch.long).to(args.device),
                        torch.tensor(observation.edge_features, dtype=torch.float32).to(args.device),
                        torch.tensor(observation.variable_features, dtype=torch.float32).to(args.device),
                    ],
                    torch.tensor([[action_index]], dtype=torch.long).to(args.device),
                    torch.tensor(action_set, dtype=torch.long).to(args.device),
                    torch.tensor(len(action_set), dtype=torch.long).to(args.device)                
                )
            observation, action_set, reward, done, info = real_env.step(action)
            if discriminator is not None:
                reward += args.airl_coef * dis_reward.item()
            total_reward += reward
        
        reward_list.append(total_reward)
        
        scip_model = real_env.get_scip_model()

        stime = scip_model.getSolvingTime()
        nnodes = scip_model.getNNodes()
        nlps = scip_model.getNLPs()
        gap = scip_model.getGap()
        status = scip_model.getStatus()
        nnodes_list.append(nnodes)
        

    return gmean(nnodes_list), np.mean(reward_list)
