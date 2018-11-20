import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from .Util import numpy2tensor_float, numpy2tensor_long,tensor2numpy_int

 #返回 edgeId, questionId, answerId一个 batch 一个 batch
 #edgeId 对应 answerId, answerId 对应 content, 以及 vote
 #TODO: useing data loader to load data.
 #https://www.kaggle.com/ankitjha/hands-on-practical-deep-learning-with-pytorch

#TODO: idx2id 在后面怎样使用
#TODO: 测试需要加入新的点，　然后去判断最佳答案
class Minibatch(object):
    def __init__(self, G, batchsize, max_degree, content_len, **kwargs):
        self.G = G
        self.batchsize = batchsize
        self.max_degree = max_degree
        self.edges = self.G.edges(data=True)
        self.nodes = self.G.nodes(data=True)

        self.content_len = content_len
        self.id2idx_float = self.__id2idx_float()
        self.adj, self.adj_edge, self.adj_score, self.deg = self.construct_adj()
        self.val_edge = self._val_edge()
        self.train_edge = np.array(list(self.__remove_isolated(self.edges)))
        self.train_edge_count = len(self.train_edge)
        self.pro = [x[2].get('score') for x in self.train_edge]
        self.pro = numpy2tensor_float(np.array(self.pro))
                

    def __id2idx_float(self):
        array = np.array([nodeId[0] for nodeId in self.nodes])
        value = np.unique(array)
        id2idx_float = {nodeId: index for index, nodeId in enumerate(value)}
        return id2idx_float

    def __id2idx_int(self):
        id2idx_int = { int(k) : v for k, v in self.id2idx_float.items()}
        return id2idx_int

    @property
    def id2idx(self):
        return self.__id2idx_int()
    @property
    def idx2id(self):
        _id2idx = self.id2idx
        idx2id = {v:k for k, v in _id2idx.items()}
        return idx2id

    def construct_adj(self):
        #TEST: successive
        adj = np.zeros(((len(self.id2idx_float) + 1), self.max_degree), dtype=np.int64)
        adj_answer = np.zeros((len(self.nodes) + 1, self.max_degree), dtype=np.int64)
        adj_score = np.zeros((len(self.nodes) + 1, self.max_degree))
        degree = np.zeros((len(self.id2idx_float),))
        for nodeid in self.G.nodes():
            neighbors = np.array([neighbor
                for neighbor in self.G.neighbors(nodeid)])

            degree[self.id2idx_float[nodeid]] = len(neighbors)

            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)

            
            for index, neigh_node in enumerate(neighbors):
                adj_answer[self.id2idx_float[nodeid], index] = int(self.G[nodeid][neigh_node]['a_id'])
                adj_score[self.id2idx_float[nodeid], index] = self.G[nodeid][neigh_node]['score']
            #STUPID: translate float into int
            adj[self.id2idx_float.get(nodeid), :] = np.array([int(neigh) for neigh in neighbors])

        return adj, adj_answer, adj_score, degree
    
    def __remove_isolated(self, edge_list):
        #TEST: success
        new_edge_list = []
        missing = 0
        for edge in edge_list:
            n1 = edge[0]
            n2 = edge[1]
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            #TODO:设置 val 性质, 准备删除一些边将这些边作为 evaluation 集合
            if (self.deg[self.id2idx_float.get(n1)] == 0 or self.deg[self.id2idx_float.get(n2)] == 0) \
                    or (self.G[n1][n2]['train_removed']):
                continue
            else:
                new_edge_list.append(edge)
        print("Unexpected missing:", missing)
        return new_edge_list

    #largevis 根据权重随机选择边
    def next_mini_batch(self):
        edgeIds = torch.multinomial(self.pro, self.batchsize, replacement=True)
        edges = self.train_edge[tensor2numpy_int(edgeIds)]
        return self.__edgeIdNodeId(edges)

    # #edge index
    # def __getitem__(self, index):
    #     edges = self.train_edge[index]
    #     return self.__edgeIdNodeId(edges)
    # def __len__(self):
    #     return self.train_edge_count

    def __edgeIdNodeId(self, edges):
        #TEST: success
        #we append userId to question, answer id; so userId > quesitonId
        questio_node = numpy2tensor_long(np.array([min(x[0], x[1]) for x in edges]))
        #puzzle convert to 32 tensor
        user_node = numpy2tensor_long(np.array([max(x[0], x[1])  for x in edges]))
        answe_edge = numpy2tensor_long(np.array([x[2].get('a_id') for x in edges]))
        edge_score = numpy2tensor_float(np.array([x[2].get('score') for x in edges]))
        #original id
        return questio_node, user_node, answe_edge, edge_score
        
    def _val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]
    
    def val_mini_batch(self):
        return self.__edgeIdNodeId(self.val_edge)
