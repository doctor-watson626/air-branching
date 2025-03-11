import random
import numpy as np
from operator import itemgetter
from observation import BipartiteNodeObs
import copy
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action_set, action, reward, next_state, action_set_next, done, info):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, action_set, reward, next_state, action_set_next, done, info
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self):
        data = random.choice(self.buffer)

        return {
            "obs":data[0],
            "action":data[1],
            "action_set":data[2],
            "reward":data[3],
            "obs_next":data[4],
            "action_set_next":data[5],
            "done":data[6], 
            "info":data[7]
        }



    def sample_batch(self, batch_size):
        if len(self.buffer) == 0:
            return None
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))

        
        # state 
        constraint_features = np.vstack(
            [one_data[0].constraint_features for one_data in batch]
        )

        edge_index_list = [copy.deepcopy(one_data[0]).edge_indices for one_data in batch]
        for i in range(len(edge_index_list)):
            for j in range(i):
                one_data = batch[j]
                edge_index_list[i][0, :] += one_data[0].constraint_features.shape[0]
                edge_index_list[i][1, :] += one_data[0].variable_features.shape[0]
        edge_index = np.hstack(edge_index_list)
        edge_attr = np.vstack(
            [one_data[0].edge_features for one_data in batch]
        )
        variable_features = np.vstack(
            [one_data[0].variable_features for one_data in batch]
        )

        state = BipartiteNodeObs(
            constraint_features, edge_index, edge_attr, variable_features
        )

        # action
        action = np.vstack(
            [one_data[1] for one_data in batch]
        )

        action_set_list = [one_data[2] for one_data in batch]
        for i in range(len(action_set_list)):
            for j in range(i):
                one_data = batch[j]
                action_set_list[i] = action_set_list[i] + one_data[0].variable_features.shape[0]
            
        action_set = np.hstack(action_set_list)

        # action_set_len
        action_set_len = np.hstack(
            [len(one_data[2]) for one_data in batch]
        )

        # reward
        reward = np.hstack(
            [one_data[3] for one_data in batch]
        )

        # state_next
        constraint_features = np.vstack(
            [one_data[4].constraint_features for one_data in batch]
        )
        edge_index = np.hstack(
            [one_data[4].edge_indices for one_data in batch]
        )
        edge_attr = np.vstack(
            [one_data[4].edge_features for one_data in batch]
        )
        variable_features = np.vstack(
            [one_data[4].variable_features for one_data in batch]
        )

        next_state = BipartiteNodeObs(
            constraint_features, edge_index, edge_attr, variable_features
        )

        action_set_next_list = [one_data[5] for one_data in batch]
        for i in range(len(action_set_next_list)):
            for j in range(i):
                one_data = batch[j]
                action_set_next_list[i] = action_set_next_list[i] + one_data[4].variable_features.shape[0]
            
        action_set_next = np.hstack(action_set_next_list)

        action_set_next_len = np.hstack(
            [len(one_data[5]) for one_data in batch]
        )

        done = np.hstack(
            [one_data[6] for one_data in batch]
        )

        variable_num_list = np.hstack(
            [one_data[4].variable_features.shape[0] for one_data in batch]
        )
        
        return {
            "obs":state,
            "action":action,
            "action_set":action_set,
            "action_set_len":action_set_len,
            "reward":reward,
            "obs_next":next_state,
            "action_set_next":action_set_next,
            "action_set_next_len":action_set_next_len,
            "done":done,
            "variable_num_list":variable_num_list
        }

    def return_all(self):
        return self.buffer
    
    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)

class PPOReplayBuffer:
    def __init__(self,args):
        self.buffer = []
        self.args = args
        
    def clear(self):
        self.buffer = []

    def push(self, state, action_set, action, reward, next_state, action_set_next, done, info, log_action_prob):
        self.buffer.append(
            [state, action, action_set, reward, next_state, action_set_next, done, info, log_action_prob]
        )

    def sample_one_batch(self, batch_size):
        if len(self.buffer) == 0:
            return None
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        res = {}
        res["constraint_features"] = np.vstack(
                [one_data[0].constraint_features for one_data in batch]
            )

        edge_index_list = [copy.deepcopy(one_data[0]).edge_indices for one_data in batch]
        for i in range(len(edge_index_list)):
            for j in range(i):
                one_data = batch[j]
                edge_index_list[i][0, :] += one_data[0].constraint_features.shape[0]
                edge_index_list[i][1, :] += one_data[0].variable_features.shape[0]
        res["edge_index"] = np.hstack(edge_index_list)

        res["edge_attr"] = np.vstack(
            [one_data[0].edge_features for one_data in batch]
        )
        res["variable_features"] = np.vstack(
            [one_data[0].variable_features for one_data in batch]
        )

        res["action"] = np.vstack(
            [one_data[1] for one_data in batch]
        )

        action_set_list = [one_data[2] for one_data in batch]
        for i in range(len(action_set_list)):
            for j in range(i):
                one_data = batch[j]
                action_set_list[i] = action_set_list[i] + one_data[0].variable_features.shape[0]
        
        action_set_ori = [one_data[2] for one_data in batch]
        res["action_set_ori"] = action_set_ori
        res["action_set"] = np.hstack(action_set_list)

        # action_set_len
        res["action_set_len"] = np.hstack(
            [len(one_data[2]) for one_data in batch]
        )

        # reward
        res["reward"] = np.vstack(
            [one_data[3] for one_data in batch]
        )

        # state_next
        res["next_constraint_features"] = np.vstack(
            [one_data[4].constraint_features for one_data in batch]
        )
        edge_index_list_next = [copy.deepcopy(one_data[4]).edge_indices for one_data in batch][:]
        for i in range(len(edge_index_list_next)):
            for j in range(i):
                one_data = batch[j]
                edge_index_list_next[i][0, :] = edge_index_list_next[i][0, :] + one_data[4].constraint_features.shape[0]
                edge_index_list_next[i][1, :] = edge_index_list_next[i][1, :] + one_data[4].variable_features.shape[0]
        res["next_edge_index"] = np.hstack(edge_index_list_next)

        res["next_edge_attr"] = np.vstack(
            [one_data[4].edge_features for one_data in batch]
        )
        res["next_variable_features"] = np.vstack(
            [one_data[4].variable_features for one_data in batch]
        )

        
        action_set_next_list = [one_data[5] for one_data in batch]
        for i in range(len(action_set_next_list)):
            for j in range(i):
                one_data = batch[j]
                action_set_next_list[i] = action_set_next_list[i] + one_data[4].variable_features.shape[0]
            
        res["next_action_set"] = np.hstack(action_set_next_list)

        res["next_action_set_len"] = np.hstack(
            [len(one_data[5]) for one_data in batch]
        )

        res["done"] = np.vstack(
            [one_data[6] for one_data in batch]
        )

        res["variable_num"] = batch[0][0].variable_features.shape[0]

        return res

    
    def get_batch(self, batch_size):
        batch_size = self.args.batch_size
        batch_list = []
        for i in range(0, len(self.buffer), batch_size):
            batch = self.buffer[i:i + batch_size]
            res = {}
            # state 
            res["constraint_features"] = np.vstack(
                [one_data[0].constraint_features for one_data in batch]
            )
            
            edge_index_list = [copy.deepcopy(one_data[0]).edge_indices for one_data in batch]
            for i in range(len(edge_index_list)):
                for j in range(i):
                    one_data = batch[j]
                    edge_index_list[i][0, :] += one_data[0].constraint_features.shape[0]
                    edge_index_list[i][1, :] += one_data[0].variable_features.shape[0]
            res["edge_index"] = np.hstack(edge_index_list)

            res["edge_attr"] = np.vstack(
                [one_data[0].edge_features for one_data in batch]
            )
            res["variable_features"] = np.vstack(
                [one_data[0].variable_features for one_data in batch]
            )


            # action
            res["action"] = np.vstack(
                [one_data[1] for one_data in batch]
            )

            action_set_list = [one_data[2] for one_data in batch]
            for i in range(len(action_set_list)):
                for j in range(i):
                    one_data = batch[j]
                    action_set_list[i] = action_set_list[i] + one_data[0].variable_features.shape[0]
            
            action_set_ori = [one_data[2] for one_data in batch]
            res["action_set_ori"] = action_set_ori
            res["action_set"] = np.hstack(action_set_list)

            # action_set_len
            res["action_set_len"] = np.hstack(
                [len(one_data[2]) for one_data in batch]
            )

            # reward
            res["reward"] = np.vstack(
                [one_data[3] for one_data in batch]
            )

            # state_next
            res["next_constraint_features"] = np.vstack(
                [one_data[4].constraint_features for one_data in batch]
            )

            edge_index_list_next = [copy.deepcopy(one_data[4]).edge_indices for one_data in batch][:]
            for i in range(len(edge_index_list_next)):
                for j in range(i):
                    one_data = batch[j]
                    edge_index_list_next[i][0, :] = edge_index_list_next[i][0, :] + one_data[4].constraint_features.shape[0]
                    edge_index_list_next[i][1, :] = edge_index_list_next[i][1, :] + one_data[4].variable_features.shape[0]
            res["next_edge_index"] = np.hstack(edge_index_list_next)

            res["next_edge_attr"] = np.vstack(
                [one_data[4].edge_features for one_data in batch]
            )
            res["next_variable_features"] = np.vstack(
                [one_data[4].variable_features for one_data in batch]
            )

            
            action_set_next_list = [one_data[5] for one_data in batch]
            for i in range(len(action_set_next_list)):
                for j in range(i):
                    one_data = batch[j]
                    action_set_next_list[i] = action_set_next_list[i] + one_data[4].variable_features.shape[0]
                
            res["next_action_set"] = np.hstack(action_set_next_list)

            res["next_action_set_len"] = np.hstack(
                [len(one_data[5]) for one_data in batch]
            )

            res["done"] = np.vstack(
                [one_data[6] for one_data in batch]
            )
            

            batch_list.append(res)

        return batch_list