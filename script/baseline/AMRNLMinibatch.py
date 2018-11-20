import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from  script.Util import numpy2tensor_long, numpy2tensor_float
from sklearn.model_selection import train_test_split

class Minibatch(object):
    def __init__(self, data, content_size, config):
        '''

        :param data: quesiton-answer-user-vote
        :param content_size: length of content
        '''
        self.quesiton_group = data.groupby("QuestionId")
        self.question_id = self.quesiton_group.groups
        self.train_ques, self.test_queis = train_test_split(self.question_id, test_size=0.33, random_state=42)
        self.config = config
        self.content_size = content_size
        self.batchnum = 0
        self.testbachnum=0

    @property
    def test_len(self):
        return len(self.test_queis)
    def next_minibatch(self):
        if(self.batchnum > len(self.train_ques)):
            self.batchnum = 0
        start_idx = self.batchnum
        self.batchnum += 1
        question_id = self.question_id[start_idx]
        return self.creat_dict(question_id)

    def creat_dict(self,question_id):
        m = self.quesiton_group.get_group(question_id)
        # ratio of vote
        ##all transported from pandas to numpy
        # TODO: performance--use List
        sorted_m = m.sort_values(by=['score'])
        answer_id_list = sorted_m['a_id'].values
        user_id_list = sorted_m['u_id'].values - self.content_size
        answer_vote_list = sorted_m['score'].values
        # padding to have the same length
        return numpy2tensor_long(question_id), numpy2tensor_long(user_id_list), \
               numpy2tensor_long(answer_id_list),\
               numpy2tensor_float(answer_vote_list)

    def evaluate_next_minibatch(self):
        if (self.testbachnum > len(self.test_queis)):
            self.testbachnum = 0
        start_idx = self.testbachnum
        self.testbachnum+= 1
        question_id = self.question_id[start_idx]
        return self.creat_dict(question_id)




