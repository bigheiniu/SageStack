import time

from .aggregators import MeanAggregator
import torch.nn as nn
import torch
from torch.nn import init, LSTM
from torch.autograd import Variable
from collections import namedtuple
import numpy as np
from .function import UserAnswer, QuestionAnswer, Score, NegScore
from .Util import numpy2tensor_float, numpy2tensor_long, numpy2tensor_int, tensor2numpy_int

SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])


class UnSupervisedGraphSage(nn.Module):
    # supervised 和 unsupervise 区别在于损失函数
    def __init__(self,G,
                 content_embedd, len_embed, word_embed, config,
                 layer_infos,
                 content_size,  # minibatch 那边要产生
                 question_size,
                 user_size,
                 deg,
                idx2id
                 ):
        super(UnSupervisedGraphSage, self).__init__()

        #self.embedd

        self.content_embed = nn.Embedding(content_embedd.shape[0], content_embedd.shape[1])
        self.content_embed.weight = nn.Parameter(numpy2tensor_int(content_embedd),requires_grad = False)


        self.word_embed = nn.Embedding(word_embed.shape[0], word_embed.shape[1])
        self.word_embed.weight = nn.Parameter(numpy2tensor_float(word_embed),requires_grad=False)
        # for pad order sort
        self.content_len_embed = nn.Embedding(len_embed.shape[0], 1)
        self.content_len_embed.weight = nn.Parameter(numpy2tensor_int(len_embed), requires_grad=False)
        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222
        self.user_embed = nn.Embedding(user_size, word_embed.shape[1])# 需要初始化, 然后是能够训练
        init.xavier_uniform(self.user_embed.weight)
        self.user_embed.weight = nn.Parameter(self.user_embed.weight)



        self.config = config
        self.batch_size = self.config.batch_size
        #  question-answer lstm model to generate vector
        self.lstm = LSTM(self.config.lstm_input_size, self.config.lstm_hidden_size, batch_first=True,
                         dropout = self.config.drop_out)

        self.user_answer = UserAnswer()
        self.question_answer = QuestionAnswer()
        self.score = Score()
        self.neg_score = NegScore()

        self.layer_infos = layer_infos
        self.dims = [word_embed.shape[1]]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.aggregators = self._init_agg()
        self.deg_question = deg[0:question_size]
        self.deg_user = deg[question_size:]
        self.content_size = content_size
        self.idx2id = idx2id
        self.G = G

    def _init_agg(self):
        aggregators = []
        for layer in range(2):
                # aggregator at current layer
            if layer == 0:
                aggregator = MeanAggregator(
                    self.dims[layer+1],
                    output_dim=self.config.dim_1,
                    act='relu'
                )
                if self.config.on_gpu:
                    aggregator.cuda()
            else:
                aggregator = MeanAggregator(
                        self.dims[layer+1],
                        output_dim=self.config.dim_1,
                        act='none'
                    )
                if self.config.on_gpu:
                    aggregator.cuda()
            aggregators.append(aggregator)
        return aggregators

    def _init_lstm_hidden(self,new_batch_size):
        hiddena = torch.zeros(1, new_batch_size, self.config.lstm_hidden_size)
        hiddenb = torch.zeros(1, new_batch_size, self.config.lstm_hidden_size)

        if self.config.on_gpu:
            hiddena = hiddena.type(torch.cuda.FloatTensor)
            hiddenb = hiddenb.type(torch.cuda.FloatTensor)
        else:
            hiddena = hiddena.type(torch.FloatTensor)
            hiddenb = hiddenb.type(torch.FloatTensor)
        hiddena = Variable(hiddena)
        hiddenb = Variable(hiddenb)
        return hiddena, hiddenb
        

    
    
    #根据输入的 node -> k-1 邻接节点 -> k-2层邻接节点 => 
    #然后再将生成的 list 返回给agg
    #layerinfo 跟 tensorflow 里面一样都是
    # quesiton 和 user 分别进入, 均为tensor
    def _sample(self, inputs, layer_infos, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        samples_edge = []
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]

        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node, answerId = sampler((samples[k], layer_infos[t].num_samples))
            # 将 reshape 修改成 view
            samples.append(node.view(support_size * batch_size))
            samples_edge.append(answerId.view(support_size * batch_size))
            # samples_edge_score.append(answer_score.view(support_size * batch_size, -1))
            support_sizes.append(support_size)
        
        sample_dic = {}
        sample_dic["samples"] = samples
        sample_dic["samples_edge_answer"] = samples_edge
        # sample_dic["samples_edge_answer_score"] = samples_edge_score
        sample_dic["support_sizes"] = support_sizes
        return sample_dic

    # questionId -> [word Index] -> [[word-vec] [word-vec] [word-vec]]
    # return [[word-vector] [word-vector] '''], [padding size]
    # word_index_list 都是已经 padding 好的
    # lstm 模型融合 + pack
    def _lstm_embedd(self, batch_id, output_size=None):
        batch_id = numpy2tensor_long(batch_id)
        try:
            word_index_list = self.content_embed(batch_id)
        except:
            print(batch_id)
        word_length = self.content_len_embed(batch_id)
        word_length = word_length.view(-1)
        word_index_list = numpy2tensor_long(word_index_list)
        word_vector_list = self.word_embed(word_index_list)

        #padding-sort
        word_length, perm_index = word_length.sort(0, descending=True)
        input = word_vector_list[perm_index]
        embed_input_x_packed = torch.nn.utils.rnn.pack_padded_sequence(input, word_length,batch_first=True)
        new_batch_size = list(input.size())[0]
        self.lstm_hidden = self._init_lstm_hidden(new_batch_size)
        self.lstm.zero_grad()
        output, _ = self.lstm(embed_input_x_packed, self.lstm_hidden)
        pack_output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        index, _ = perm_index.sort(0, descending=False)
        #average pooling for hidden state
        pack_output = torch.mean(pack_output,dim=1)
        return pack_output[index]

    #假设就只有两层
    def _aggregate(self, sample_dic, node_kind, num_samples, aggregators=None):
        node = sample_dic['samples']
        answers = sample_dic['samples_edge_answer']
        # edges_score = sample_dic['samples_edge_answer_score']
        support_size = sample_dic['support_sizes']

        #ordinary -> next_layer -> last_layer
        tim1 = time.time()
        if(node_kind == "question"):
            hidden = []
            hidden.append(self._lstm_embedd(node[0]))
            hidden.append(self.user_embed(numpy2tensor_long(node[1] - self.content_size)))
            hidden.append(self._lstm_embedd(node[2]))
            
        else:
            hidden = [self.user_embed(numpy2tensor_long(node[0] - self.content_size))]
            hidden.append(self._lstm_embedd(node[1]))
            hidden.append(self.user_embed(numpy2tensor_long(node[2] - self.content_size)))

        answer_hidden = [self._lstm_embedd(answer) for answer in answers]
        tim2 = time.time()
        # print("lstm time {}".format(tim2-tim1))
        old_agg = aggregators is None
        if old_agg:
            aggregators = self.aggregators
        #model init
        for layer in range(len(num_samples)):
            aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                neigh_dims = [self.batch_size * support_size[hop],
                              num_samples[len(num_samples) - hop - 1], 
                              self.dims[layer]]
                if(( node_kind == "quesiton" and  hop == 0 ) or (node_kind == 'user' and hop == 1)):
                    h = aggregator(hidden[hop],
                                self.user_answer(hidden[hop + 1], answer_hidden[hop]).view(neigh_dims))
                else:
                    h = aggregator(hidden[hop], self.question_answer(hidden[hop+1], answer_hidden[hop]).view(neigh_dims))
                next_hidden.append(h)
            hidden = next_hidden
        tim3 = time.time()
        # print("agg time is {}".format(tim3 - tim2))
        # if(tim3+tim1-2*tim2 > 0):
            # print("agg time longer {}".format(tim1+tim3-2*tim2))
        return hidden[0]

    def forward(self, questionNodeId, userNodeId):
        #node embedding by their neighbors
        #use sampler to find their neighbors
        sample_dic1 = self._sample(questionNodeId, self.layer_infos)
        sample_dic2 = self._sample(userNodeId, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        ouput_question = self._aggregate(sample_dic1,"question",num_samples)
        output_user = self._aggregate(sample_dic2,"user", num_samples)
        return (ouput_question, output_user)

    def loss(self, question_node_id, user_node_id, answer_edge_id):
        # tim1 = time.time()
        question_node_vec, user_node_vec = self.forward(question_node_id, user_node_id)
        # tim2 = time.time()
        # print("?question, user embed {}".format(tim2 - tim1))
        answer_edge_vec = self._lstm_embedd(answer_edge_id,'answer')
        
        #nagative sampling
        loss_value = torch.empty(self.batch_size+1)
        if self.config.on_gpu:
            loss_value = loss_value.type(torch.cuda.FloatTensor)
        else:
            loss_value = loss_value.type(torch.FloatTensor)
        for i in range(self.batch_size):
            neg_user_id = torch.multinomial(torch.pow(numpy2tensor_float(self.deg_user), 0.75), self.config.neg_sample_size)
            neg_question = torch.multinomial(torch.pow(numpy2tensor_float(self.deg_question), 0.75), self.config.neg_sample_size)
            neg_question = tensor2numpy_int(neg_question)
            neg_question = np.array([self.idx2id.get(q) for q in neg_question])

            # neg_user_id = tensor2numpy_int(neg_user_id)
            # neg_user_id = numpy2tensor_long(np.array([self.idx2id.get(u + self.content_size) for u in neg_user_id]))
            neg_user_id = neg_user_id

            # neg_question_vec, neg_user_vec = self.forward(neg_question, neg_user_id)
            neg_question_vec = self._lstm_embedd(neg_question)
            neg_user_vec = self.user_embed(neg_user_id)
            neg_score = torch.sum(self.neg_score(question_node_vec[i], neg_user_vec)) + torch.sum(self.neg_score(user_node_vec[i], neg_question_vec))
            loss_value[i] = -1.0 * neg_score

        pos_los = self.score(question_node_vec[i], user_node_vec[i], answer_edge_vec[i])
        loss_value[self.batch_size] = pos_los
        #TODO:为了简化, 就假设 negative 选中边的人没有回答问题
        loss_value = torch.sum(loss_value)
        tim3 = time.time()
        # print("negative samp {}".format(tim3 - tim2))
        return loss_value

    def evaluate(self):
        # use new embedding vector to find the best answer, and compare it to previous edge
        # best answer calculation
        # best user who give the best answer

        # best user, answer, question pair
        quesitons = {n for n, d in self.G.nodes(data=True) if d['bipartite'] == 0}
        count = 0
        for question in quesitons:
            if (question == float(self.content_size)):
                continue
            users = np.array([neighbor
                for neighbor in self.G.neighbors(question)])
            user_score = {user: self.G[question][user]['score'] for user in users}
            user_answer = {user: self.G[question][user]['a_id'] for user in users}
            target = int(sorted(user_score.items(), key=lambda x: -x[1])[0][0]) - self.content_size
            question_id = int(question)
            question_vec = self._lstm_embedd(np.array([question_id]))
            users_id = np.array([int(user) - self.content_size for user in users])
            users_id = numpy2tensor_long(users_id)
            answer_id = np.array([int(user_answer.get(user)) for user in users])
            user_vec = self.user_embed(users_id)
            answer_vec = self._lstm_embedd(answer_id)
            score = self.score(question_vec, user_vec, answer_vec, True)
            index = torch.argmax(score)
            if(users_id[index] == target ):
                count = count + 1
        return (1. * count) / (len(quesitons) * 1.0)


            
        
        









