import torch.nn as nn
import torch
from torch.nn import init
from torch.autograd import Variable
import torch.optim
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict




from script.minibatch import Minibatch
from script.neigh_samplers import UniformNeighborSampler
from script.model import SAGEInfo, UnSupervisedGraphSage
from script.Util import loadData, load_embedding,load_config
from script.Config import config
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(G, content_embedd, word_embed, len_embed, content_size, user_size, config):
    minibatch = Minibatch(G, config.batch_size, config.max_degree, content_size)
    adj, adj_edge, adj_score, deg = minibatch.construct_adj()
    id2idx = minibatch.id2idx
    idx2id = minibatch.idx2id
    question_size = len(id2idx) - user_size
    sampler = UniformNeighborSampler(adj, adj_edge, id2idx)
    layer_infos = [
        SAGEInfo("node", sampler, config.samples_1, config.dim_1),
        SAGEInfo("node", sampler, config.samples_2, config.dim_2)]

    model = UnSupervisedGraphSage(G, content_embedd, len_embed, word_embed, config,
                                  layer_infos,
                                  content_size,
                                  question_size,
                                  user_size,
                                  deg,
                                  idx2id
                                  )
    paramer = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(paramer, lr=0.001)
    i = 0

    device = torch.device('cuda') if config.on_gpu else torch.device('cpu')
    model.to(device)
    while i < config.edge_sample:
        question_node, user_node, answe_edge, _ = minibatch.next_mini_batch()
        loss = model.loss(question_node, user_node, answe_edge)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("backward time {}".format(tim2 - tim1))
        # print("step time {}".format(tim3 - tim2))
        # if config.debug:
        #     with torch.autograd.profiler.profile() as prof:
        #         question_node, user_node, answe_edge, _ = minibatch.next_mini_batch()
        #         loss = model.loss(question_node, user_node, answe_edge)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #     print(prof)
        #     exit()
        if (i + 1) % 100 == 0:
            with torch.no_grad():
                result = model.evaluate()
            print("finish {}".format(i))
            print("score {}".format(result))

        i = i + 1
    
    #训练完成进行 evaluation

    # minimize the loss




if __name__ == '__main__':
    config = load_config()
    file_dir_list = config.file_dir_list
    G, content_len, user_len,content= loadData(file_dir_list)
    content_embedd, len_embed, word_embed= load_embedding(content)

    train(G,content_embedd, word_embed,len_embed, content_len, user_len, config)



