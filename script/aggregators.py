import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    #TEST: success
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings with batch normalization
    """
    def __init__(self, input_dims,
    output_dim, act='relu', bias=True):
        """
        Initializes the aggregator f or a specific graph.
d
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.act = act
        self.linear = nn.Linear(input_dims, output_dim, bias)
        self.bn = nn.BatchNorm1d(self.output_dim)

    
    #直接进行加法就完事

    def forward(self, self_vecs, neigh_vecs,drop_pro=0.):
        drop_out = nn.Dropout(p = drop_pro)
        neigh_vecs = drop_out(neigh_vecs)
        self_vecs = drop_out(self_vecs)
        t = torch.unsqueeze(self_vecs, 1)
        t1 = torch.cat((neigh_vecs,t), 1)
        means = torch.mean(t1, 1)
        result = self.linear(means)
        if self.act == 'relu':
            result = F.relu(result)
        elif self.act == 'sigmoid':
            result = F.sigmoid(result)
        elif self.act == 'tahn':
            result = F.tanh(result)
        result = self.bn(result)

        return result


class PoolAggregator(nn.Module):
    # TEST: success
    """
    pooling a node's embeddings based on its neighbor
    """

    def __init__(self, input_dims,
                 output_dim, pool='mean', bias=True):
        """
        Initializes the aggregator f or a specific graph.
d
        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(PoolAggregator, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.pool = pool
        self.linear = nn.Linear(2 * input_dims, output_dim, bias)
        self.bn = nn.BatchNorm1d(self.output_dim)

    # 直接进行加法就完事

    def forward(self, self_vecs, neigh_vecs, drop_pro=0.):
        #self_vecs is 2d
        drop_out = nn.Dropout(p=drop_pro)
        neigh_vecs = drop_out(neigh_vecs)
        support_size = neigh_vecs.size()[1]
        neigh_vecs = neigh_vecs.transpose(1,2).contiugous()
        if(self.pool == 'mean'):
            neigh_vecs = F.avg_pool1d(neigh_vecs,support_size)
        else:
            neigh_vecs = F.max_pool1d(neigh_vecs, support_size)
        neigh_vecs = torch.squeeze(neigh_vecs, -1)
        self_vecs = drop_out(self_vecs)

        concat_vec = torch.cat((neigh_vecs, self_vecs), 1)
        result = self.linear(concat_vec)
        if self.act == 'relu':
            result = F.relu(result)
        elif self.act == 'sigmoid':
            result = F.sigmoid(result)
        elif self.act == 'tahn':
            result = F.tanh(result)
        result = self.bn(result)

        return result