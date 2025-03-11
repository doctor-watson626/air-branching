import numpy as np
from observation import BipartiteNodeObs
import queue
import ecole


class RealEnv:
    def __init__(self,ecole_env):
        self.ecole_env = ecole_env
    
    def get_scip_model(self):
        scip_model = self.ecole_env.model.as_pyscipopt()
        return scip_model
    
    def ecole_to_gcn(self,obs, action_set, reward, done, info):
        constraint_features = obs.row_features
        edge_indices = obs.edge_features.indices.astype(np.int32)
        edge_features = np.expand_dims(obs.edge_features.values, axis=-1)
        try:
            variable_features = obs.column_features
        except:
            variable_features = obs.variable_features

        obs_gcn = BipartiteNodeObs(
            constraint_features, edge_indices, edge_features, variable_features
        )

        candidates = np.array(action_set, dtype=np.int32)
        # except:
        #     print(done)
        return obs_gcn, candidates, reward, done, info

    def reset(self,scip_model):
        obs, action_set, reward, done, info = self.ecole_env.reset(scip_model)
        if not done:
            obs, action_set, reward, done, info = self.ecole_to_gcn(obs,action_set,reward,done,info)
        return obs, action_set, reward, done, info
    
    def step(self,acton):
        obs, action_set, reward, done, info = self.ecole_env.step(acton)
        if not done:
            obs, action_set, reward, done, info = self.ecole_to_gcn(obs,action_set,reward,done,info)
        return obs, action_set, reward, done, info