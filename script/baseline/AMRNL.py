import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from script.Util import numpy2tensor_float, numpy2tensor_long, numpy2tensor_int

class AMRNL(nn.Module):
    def __init__(self, content, word_embed,len_embed,
                 user_size,config
                 ):
        '''

        :param data: question => answer-user-vote
        :param content: content_data
        '''
        super(AMRNL, self).__init__()
        self.content_embed = nn.Embedding(
            content.shape[0],
            content.shape[1]
        )
        self.config = config
        self.content_embed.weight = nn.Parameter(numpy2tensor_long(content),requires_grad = False)

        self.word_embed = nn.Embedding(word_embed.shape[0], word_embed.shape[1])
        self.word_embed.weight = nn.Parameter(numpy2tensor_float(word_embed), requires_grad=False)

        self.content_len_embed = nn.Embedding(len_embed.shape[0], 1)
        self.content_len_embed.weight = nn.Parameter(numpy2tensor_int(len_embed), requires_grad=False)

        self.user_embdding = nn.Embedding(user_size, word_embed.shape[1])
        nn.init.xavier_uniform_(self.user_embdding.weight)
        self.lstm = nn.LSTM(self.config.lstm_input_size, self.config.lstm_hidden_size, batch_first=True,
                         dropout = self.config.drop_out)
        self.constant = 0.001



    def lstm_embedding(self, batch_id):
        word_index_list = self.content_embed(batch_id)
        word_length = self.content_len_embed(batch_id)
        word_length = word_length.view(-1)
        word_vector_list = self.word_embed(word_index_list)
        #padding-sort
        #TODO: chcek rnn
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
        pack_output = torch.mean(pack_output, dim=1)
        return pack_output[index]

    def _init_lstm_hidden(self,new_batch_size):
        hiddena = torch.zeros(1, new_batch_size, self.config.lstm_hidden_size)
        hiddenb = torch.zeros(1, new_batch_size, self.config.lstm_hidden_size)

        if self.config.on_gpu:
            hiddena = hiddena.type(torch.cuda.FloatTensor)
            hiddenb = hiddenb.type(torch.cuda.FloatTensor)
        else:
            hiddena = hiddena.type(torch.FloatTensor)
            hiddenb = hiddenb.type(torch.FloatTensor)
        return hiddena, hiddenb

    def forward(self, question_id, user_ids, answer_ids, vote_scores):
        question_vec = self.lstm_embedding(question_id)
        answer_vec = self.lstm_embedding(answer_ids)
        user_vec = self.user_embdding(user_ids)
        loss = torch.mul(torch.matmul(question_vec, answer_vec.t().contiguous()),
                         torch.matmul(question_vec, user_vec.t().contiguous()))
        loss.sub_(loss[[0]])
        loss.add_(self.constant)
        F.relu(loss, inplace=True)
        vote_scores = vote_scores.unsqueeze(0)
        vote_scores = vote_scores.view(-1,1)
        vote_scores_mat = vote_scores.expand_as(user_vec)
        vote_scores = torch.div(vote_scores_mat, vote_scores)

        user_loss  = torch.div(torch.matmul(vote_scores, user_vec.t()), 2)
        loss = loss + user_loss
        loss = torch.sum(loss)
        return loss

    def evaluate(self, question_id, user_ids, answer_ids, vote_scores):
        question_vec = self.lstm_embedding(question_id)
        answer_vec = self.lstm_embedding(answer_ids)
        user_vec = self.user_embdding(user_ids)
        loss = torch.mul(torch.matmul(question_vec, answer_vec.t().contiguous()),
                         torch.matmul(question_vec, user_vec.t().contiguous()))
        loss.sub_(loss[[0]])
        if loss[loss > 0].size()[0] > 1:
            return 0
        else:
            return 1







