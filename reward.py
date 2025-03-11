import copy
import numpy as np
import random
from search_tree import SearchTree

class BinaryFathomed:
    def __init__(self, not_fathomed=-1, fathomed=0):
        self.not_fathomed = not_fathomed
        self.fathomed = fathomed

    def before_reset(self, model):
        self.prev_node = None
        self.prev_node_id = None
        self.prev_primal_bound = None
        self.init_primal_bound = None

    def extract(self, model, done):
        m = model.as_pyscipopt()

        if self.prev_node_id is None:
            # not yet started, update prev node for next step
            self.prev_node = m.getCurrentNode()
            self.tree = SearchTree(model)
            if self.prev_node is not None:
                self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
            return 0

        # update search tree with current model state
        self.tree.update_tree(model)
        
        # collect node stats from children introduced from previous branching decision
        prev_node_child_ids = [child for child in self.tree.tree.successors(self.prev_node_id)]

        # calc reward for previous branching decision
        if len(prev_node_child_ids) > 0:
            # previous branching decision did not fathom sub-tree
            closed_by_agent = False
            score = self.not_fathomed
        else:
            # previous branching decision fathomed sub-tree
            closed_by_agent = True
            score = self.fathomed

        # update tree with effect(s) of branching decision
        self.tree.tree.nodes[self.prev_node_id]['score'] = score
        self.tree.tree.nodes[self.prev_node_id]['closed_by_agent'] = closed_by_agent

        if m.getCurrentNode() is not None:
            # update stats for next step
            self.prev_node = m.getCurrentNode()
            self.prev_node_id = copy.deepcopy(self.prev_node.getNumber())
        else:
            # instance completed, no current focus node
            pass

        return score

class BinarySolved:
    def __init__(self, not_solved=-1, solved=0):
        '''Returns solved if step solved instance, otherwise returns not_solved.'''
        self.not_solved = not_solved
        self.solved = solved

    def before_reset(self, model):
        pass

    def extract(self, model, done):
        if done:
            return self.solved
        else:
            return self.not_solved


class PrimalDualGapFrac:
    '''
    Evaluates change in primal-dual gap normalised w.r.t. initial primal-dual gap.
    '''
    def __init__(self):
        pass

    def before_reset(self, model):
        self.init_gap = None

    def extract(self, model, done):
        '''Updates the internal primal-dual gap and returns the the fractional difference.'''
        m = model.as_pyscipopt()
        if self.init_gap is None:
            self.init_gap = abs(m.getDualbound() - m.getPrimalbound())
            self.gap = copy.deepcopy(self.init_gap)
            return 0
        if self.init_gap == 0:
            # was pre-solved
            return 0
        else:
            self.prev_gap = copy.deepcopy(self.gap)
            self.gap = abs(m.getDualbound() - m.getPrimalbound())
            reward = (self.prev_gap - self.gap) / self.init_gap
            return reward

class DualBoundFrac:
    '''
    Evaluates change in dual bound normalised w.r.t. initial dual bound.
    '''
    def __init__(self, sense=-1):
        '''If minimising, must set sense=-1 to incentivise largest delta in dual bound.'''
        self.sense = sense

    def before_reset(self, model):
        self.init_dual_bound = None

    def extract(self, model, done):
        '''Updates the internal dual bound and returns the fractional difference.'''
        m = model.as_pyscipopt()
        if self.init_dual_bound is None:
            self.init_dual_bound = m.getDualbound()
            self.dual_bound = m.getDualbound()
            return 0
        self.prev_dual_bound = copy.deepcopy(self.dual_bound)
        self.dual_bound = m.getDualbound()
        reward = self.sense*(self.prev_dual_bound-self.dual_bound)/self.init_dual_bound
        return reward 