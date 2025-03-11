import torch
import torch_geometric
import numpy as np
import gzip
import pickle
from pathlib import Path
import ecole

import os
from observation import ExploreThenStrongBranch


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

        mip_constraint_features,
        mip_variable_features,
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

        self.mip_constraint_features = mip_constraint_features
        self.mip_variable_features = mip_variable_features

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
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    


    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
 
        with gzip.open(self.sample_files[index], "rb") as f:
            sample = pickle.load(f)

        sample_observation, mip_observation, sample_action, sample_action_set, sample_scores = sample
        
        constraint_features = sample_observation.row_features
        edge_indices = sample_observation.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(sample_observation.edge_features.values, axis=-1)
        try:
            variable_features = sample_observation.column_features
        except:
            variable_features = sample_observation.variable_features

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = np.array(sample_action_set, dtype=np.int32)
        candidate_scores = np.array([sample_scores[j] for j in candidates])
        candidate_choice = np.where(candidates == sample_action)[0][0]


        mip_constraint_features = mip_observation.row_features
        try:
            mip_variable_features = mip_observation.column_features
        except:
            mip_variable_features = mip_observation.variable_features

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(candidates),
            len(candidates),
            torch.LongTensor([candidate_choice]),
            torch.FloatTensor(candidate_scores),

            torch.FloatTensor(mip_constraint_features),
            torch.FloatTensor(mip_variable_features),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph
    
def generate_dataset(args, instances):

    scip_parameters = {
        "separating/maxrounds": 0,
        "presolving/maxrestarts": 0,
        "limits/time": 3600,
    }

    # Note how we can tuple observation functions to return complex state information
    env = ecole.environment.Branching(
        observation_function=(
            # 完全调用strong branching
            ExploreThenStrongBranch(expert_probability=0.05), 
            ecole.observation.NodeBipartite(),
        ),
        scip_params=scip_parameters,
    )

    # This will seed the environment for reproducibility
    env.seed(args.seed)
    # instances.seed(args.seed)

    episode_counter, sample_counter = 0, 0

    if not os.path.exists("samples/{}/".format(args.instance_name)):
        os.makedirs("samples/{}/".format(args.instance_name))

    # if not os.path.exists("samples/{}/instances/".format(args.instance_name)):
    #     os.makedirs("samples/{}/instances/".format(args.instance_name))

    Path("samples/{}/".format(args.instance_name)).mkdir(exist_ok=True)
    # Path("samples/{}/instances/".format(args.instance_name)).mkdir(exist_ok=True)

    # We will solve problems (run episodes) until we have saved enough samples
    while sample_counter < args.data_max_samples:
        episode_counter += 1

        instance = next(instances)
        observation, action_set, _, done, _ = env.reset(instance)
        while not done:
            (scores, scores_are_expert), node_observation = observation
            
            # 在动作集合中，分数最高的那个action是哪个
            action = action_set[scores[action_set].argmax()]


            x = -scores[action_set]
            exp_scores = np.exp(x - np.max(x))
            action_probs = exp_scores / np.sum(exp_scores)
            bad_action = np.random.choice(action_set, p=action_probs)
            
            # Only save samples if they are coming from the expert (strong branching)
            if scores_are_expert and (sample_counter < args.data_max_samples):
                sample_counter += 1
                data = [node_observation, action, action_set, scores]
                filename = f"samples/{args.instance_name}/sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(data, f)

                bad_data = [node_observation, bad_action, action_set, scores]
                filename = f"samples/{args.instance_name}/bad_sample_{sample_counter}.pkl"

                with gzip.open(filename, "wb") as f:
                    pickle.dump(bad_data, f)

            observation, action_set, _, done, _ = env.step(action)

        # instance.write_problem(f"samples/{args.instance_name}/instances/instance_{episode_counter}.lp")
        print(f"Episode {episode_counter}, {sample_counter} samples collected so far")