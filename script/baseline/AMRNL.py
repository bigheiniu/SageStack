import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from script.Util import numpy2tensor_float, numpy2tensor_long

class AMRNL(nn.Module):
    def __init__(self, content, word_embed,
                 user_size
                 ):
        '''

        :param data: question => answer-user-vote
        :param content: content_data
        '''
        super(AMRNL, self).__init__()
        self.content_embedding = nn.Embedding(
            content.shape[0],
            content.shape[1]
        )
        self.content_embedding.weight = nn.Parameter(numpy2tensor_long(content),requires_grad = False)

        self.word_embed = nn.Embedding(word_embed.shape[0], word_embed.shape[1])
        self.word_embed.weight = nn.Parameter(numpy2tensor_float(word_embed), requires_grad=False)

        self.user_embdding = nn.Embedding(user_size, word_embed.shape[1])
        nn.init.xavier_uniform_(self.user_embed.weight)
        self.constant = 0.001


    def lstm_embedding(self, batch_id):
        try:
            word_index_list = self.content_embed(batch_id)
        except:
            print(batch_id)
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
        vote_scores_mat = vote_scores.expand_as(user_vec)
        vote_scores = torch.div(vote_scores_mat, vote_scores.t())

        user_loss  = torch.div(torch.matmul(vote_scores, user_vec), 2)
        loss = loss + user_loss
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
        else: return 1







