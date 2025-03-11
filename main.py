import ecole.instance
from reward import BinarySolved, PrimalDualGapFrac,RetroBranching, BinaryFathomed, DualBoundFrac
import ecole
# from pyscipopt import Model, quicksum
import numpy as np

from replay_buffer import PPOReplayBuffer
from utils import evaluate_agent
import argparse
from env import RealEnv, DFSBranchingEnv

from torch.utils.tensorboard import SummaryWriter

import torch

import gc
import tracemalloc
import logging

from pathlib import Path
import itertools
import os
import shutil  
from agent.ppo import PPO
from dataset import generate_dataset
from torch.utils.tensorboard import SummaryWriter
import gzip
import pickle

from agent.airl import AIRL, train_airl, get_dataloader, pretrain_airl
import random
import torch.nn.functional as F

if __name__ == '__main__':

    # 命令行获取参数
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma in rl')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    parser.add_argument('--batch_size',type=int,default=32, help='batch size')
    parser.add_argument('--D_env_min_num',type=int,default=2000,help='Minimum capacity of D_env to collect this much data before training')
    parser.add_argument('--model_save_cycle',type=int,default=2000, help='How many times to store the model per train')
    parser.add_argument('--evaluate_cycle',type=int,default=20, help='evaluate cycle (episode)')

    parser.add_argument('--gpu',type=int,default=3, help='gpu id')
    parser.add_argument('--alg',type=str,default='ppo', help='agent')

    parser.add_argument('--run_id',type=int,default=100, help='Run number')
    
    parser.add_argument('--train_instance_size',nargs=2,type=int,default=[100,500], help='The size of the instance used for training')
    parser.add_argument('--evaluate_instance_size',nargs=2,type=int,default=[100,500], help='The size of the instance used for evaluation')

    parser.add_argument('--instance_name',type=str,default='ca',help='Problem Name')
    parser.add_argument('--max_episode_num',type=int, default=1000,help="Each time you evaluate, you take the average of several episodes")
    parser.add_argument('--evaluate_num',type=int, default=20,help="")

    parser.add_argument('--seed',type=int, default=0,help="random seed")
    parser.add_argument('--is_generate_samples', action='store_true', help='Indicates that we need to generate the strong branching dataset')
    parser.add_argument('--is_airl', action='store_true', help='Whether airl is used or not')

    parser.add_argument('--airl_train_epoch',type=int, default=10,help="The number of times airl is trained in one session")
    parser.add_argument('--airl_pretrain_epoch',type=int, default=10,help="The number of times airl is pretrained in one session")

    parser.add_argument('--ppo_train_epoch',type=int, default=10,help="Number of batches for PPO training")
    parser.add_argument('--data_max_samples',type=int, default=1000,help="The number of samples collected")

    parser.add_argument('--is_airl_init', action='store_true', help='Whether to add init')
    parser.add_argument('--time_limit',type=int, default=600,help="Time limit")

    parser.add_argument('--airl_coef',type=float, default=0.1,help="The coefficient of reward for airl")
    parser.add_argument('--entropy_coef',type=float, default=0.1,help="The entropy coefficient of ppo")

    parser.add_argument('--reward_type',type=str,default='retro',help='Rewards used')




    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    argsDict = args.__dict__
    with open('{}_{}_setting.txt'.format(args.alg,args.run_id), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
    f.close()
    
    # tensorboard 
    writer = SummaryWriter('./logs/{}'.format(args.run_id))
    
    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": args.time_limit,
    }

    if args.reward_type == 'binary_fathomed':
        reward_function = BinaryFathomed()
    elif args.reward_type == 'primal_dual_gap_frac':
        reward_function = PrimalDualGapFrac()
    elif args.reward_type == 'binary_solved':
        reward_function = BinarySolved()
    elif args.reward_type == 'constant':
        reward_function = None
    elif args.reward_type == 'dual_bound':
        reward_function = DualBoundFrac()
    else:
        raise NotImplementedError("reward type not implemented")

   
    ecole_env = ecole.environment.Branching(
        observation_function = ecole.observation.NodeBipartite(),
        scip_params=scip_parameters, 
        reward_function=reward_function,
    )



    real_env = RealEnv(ecole_env)
    agent = PPO(args)

    D_env = PPOReplayBuffer(args)

    discriminator = AIRL(args).to(args.device)
        
    if args.instance_name == 'setcover':
        dir_name = "data/instances/setcover/train_{}r_{}c_0.05d".format(
            args.train_instance_size[0],args.train_instance_size[1]
        )
        train_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        np.random.shuffle(train_instance_files)
        instances = itertools.cycle(train_instance_files)
        
        dir_name = "data/instances/setcover/valid_{}r_{}c_0.05d".format(
            args.evaluate_instance_size[0],args.evaluate_instance_size[1]
        )
        valid_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        fixed_instances = itertools.cycle(np.random.choice(valid_instance_files, args.evaluate_num))

    elif args.instance_name == 'ca':

         
        dir_name = "data/instances/cauctions/train_{}_{}".format(
            args.evaluate_instance_size[0],args.evaluate_instance_size[1]
        )
        train_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        np.random.shuffle(train_instance_files)
        instances = itertools.cycle(train_instance_files)
        
        dir_name = "data/instances/cauctions/valid_{}_{}".format(
            args.evaluate_instance_size[0],args.evaluate_instance_size[1]
        )
        valid_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        fixed_instances = itertools.cycle(np.random.choice(valid_instance_files, args.evaluate_num))

    elif args.instance_name == 'indset':


        dir_name = "data/instances/indset/train_{}_4".format(args.train_instance_size[0])
        train_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        np.random.shuffle(train_instance_files)
        instances = itertools.cycle(train_instance_files)
        
        dir_name = "data/instances/indset/valid_{}_4".format(args.evaluate_instance_size[0])
        valid_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        fixed_instances = itertools.cycle(np.random.choice(valid_instance_files, args.evaluate_num))


    elif args.instance_name == 'cfl':
        dir_name = f"data/instances/ufacilities/train_{args.train_instance_size[0]}_{args.train_instance_size[1]}_5"
        train_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        np.random.shuffle(train_instance_files)
        instances = itertools.cycle(train_instance_files)
        
        dir_name = f"data/instances/ufacilities/valid_{args.evaluate_instance_size[0]}_{args.evaluate_instance_size[1]}_5"
        valid_instance_files = [str(path) for path in Path(f"{dir_name}/").glob("instance_*.lp")]
        fixed_instances = itertools.cycle(np.random.choice(valid_instance_files, args.evaluate_num))

       
   
    else:
        raise NotImplementedError("instance not implemented")

    if args.is_generate_samples:
        generate_dataset(args, instances)

    

    strong_loader, pseudo_loader = get_dataloader(args)
    strong_loader = itertools.cycle(strong_loader)
    pseudo_loader = itertools.cycle(pseudo_loader)

    if args.is_airl_init:
        loss = pretrain_airl(discriminator, args)
        writer.add_scalar('airl_loss', loss, 0)

    for i in range(args.max_episode_num):

        print("episode {}".format(i))
        
        origin_transitions = []
        observation, action_set, done, info = None, None, True, None

        while done:
            scip_model = next(instances)
            observation, action_set, _, done, info = real_env.reset(scip_model)
        
        print("agent interacts with environment")
        step_cnt = 0

        while not done:
            assert observation != None
            obs_now = observation
            action_set_now = action_set
            info_now = info

            action, log_action_prob = agent.choose_action(obs_now,action_set_now)

         
            observation, action_set, reward, done, info = real_env.step(action)
            if args.reward_type == "constant":
                reward = 0

            
            obs_next = observation
            action_set_next = action_set

            
            if not done:
                origin_transitions.append(
                # D_env.push(
                    [obs_now,action_set_now,action,reward,obs_next,action_set_next,done,None,log_action_prob]
                )
            else:
                
                origin_transitions.append(
                # D_env.push(
                    [obs_now,action_set_now,action,reward,obs_now,np.array([]),done,None, log_action_prob]
                )
            step_cnt += 1
        
        print(len(origin_transitions))

        if len(origin_transitions) == 0:
            print("D_env.buffer is empty, skip the training")
            continue
      
        for transition in origin_transitions:
            D_env.push(*transition)
        
        D_env.buffer = D_env.buffer[1:]

        if len(D_env.buffer) == 0:
            print("D_env.buffer is empty, skip the training")
            continue
        print("start training")
        if args.is_airl:
            loss = train_airl(discriminator, D_env, agent, strong_loader, pseudo_loader, args)
            writer.add_scalar('airl_loss', loss, i+1)

            for index, transition in enumerate(D_env.buffer[:]):
                log_action_prob = transition[8]
                state = transition[0]
                action = transition[1]

                action_set = transition[2]
                action_index = np.where(action_set == action)[0][0]

                dis_reward = discriminator.get_reward(
                    torch.tensor(log_action_prob, dtype=torch.float32).view(1,1).to(args.device),
                    [
                        torch.tensor(state.constraint_features, dtype=torch.float32).to(args.device),
                        torch.tensor(state.edge_indices, dtype=torch.long).to(args.device),
                        torch.tensor(state.edge_features, dtype=torch.float32).to(args.device),
                        torch.tensor(state.variable_features, dtype=torch.float32).to(args.device),
                    ],
                    torch.tensor([[action_index]], dtype=torch.long).to(args.device),
                    torch.tensor(action_set, dtype=torch.long).to(args.device),
                    torch.tensor(len(action_set), dtype=torch.long).to(args.device)                
                )
                transition[3] = transition[3] + args.airl_coef * dis_reward.item()
                D_env.buffer[index] = transition

      
        all_batch_list = D_env.get_batch(args.batch_size)
        ppo_train_epoch = min(args.ppo_train_epoch, len(all_batch_list))

        
        batch_list = random.sample(all_batch_list, ppo_train_epoch)
        for batch in batch_list:
            agent.train(batch)
       

        D_env.clear()

        if i % args.evaluate_cycle == 0 and i != 0:
            nnodes, evaluate_reward = evaluate_agent(agent, args, fixed_instances,args.evaluate_num, discriminator)
            writer.add_scalar('nnodes_episode', nnodes, i)
            writer.add_scalar('reward_episode', evaluate_reward, i)
            agent.save_current_model()
    
    writer.close()
    agent.save_model()