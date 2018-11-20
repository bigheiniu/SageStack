from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from .Util import numpy2tensor_long, tensor2numpy_int, numpy2tensor_int
import numpy as np

"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(torch.nn.Module):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, adj_answer, id2idx):
        super(UniformNeighborSampler, self).__init__()
        # outsize init do embedding
        self.x_ = adj_info.shape[0]
        self.y_ = adj_info.shape[1]
        self.adj_info = nn.Embedding(self.x_,self.y_)
        self.adj_info.weight = nn.Parameter(numpy2tensor_long(adj_info),False)
        self.adj_answer = nn.Embedding(self.x_, self.y_)
        self.adj_answer.weight = nn.Parameter(numpy2tensor_long(adj_answer),False)
        self.id2idx = id2idx


    def forward(self, inputs):
        '''
        edge selection => edge probability
        '''
        #PROBLEM: id2idx出现断层现象
        #PROBLEM: quesiton, userId confiliction => solve minibatch function
        ids, num_samples = inputs
        # convert ids to idx => get the column of adjancy matrix
        ids = tensor2numpy_int(ids)
        ids = np.array([self.id2idx.get(id) for id in ids])
        try:
            th = ids
            ids = numpy2tensor_long(ids)
        except:
            print(th)
        adj_lists = self.adj_info(ids)
        adj_answerId_lists = self.adj_answer(ids)
        index = torch.randperm(self.y_)
        index = index[0: num_samples]
        adj_lists = adj_lists[:, index]
        adj_answerId_lists = adj_answerId_lists[:, index]
        return adj_lists, adj_answerId_lists
