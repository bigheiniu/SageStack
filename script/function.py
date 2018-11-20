import torch 
import torch.nn as nn


class UserAnswer(nn.Module):
    def __init__(self):
        super(UserAnswer,self).__init__()
    
    def forward(self, user, answer):
        return torch.mul(user, answer)
    
class QuestionAnswer(nn.Module):
    def __init__(self):
        super(QuestionAnswer,self).__init__()
    
    def forward(self, question, answer):
        return torch.mul(question, answer)

class Score(nn.Module):
    # q_v * u_v + q_v * M * a_v
    def __init__(self):
        super(Score,self).__init__()

    def forward(self, question, user, answer, batch=False):
        if(batch):
            return torch.matmul(user, question.view(-1,1)) + torch.matmul(answer, question.view(-1,1))
        else:
            return torch.matmul(question, user) + torch.matmul(question,answer)


class NegScore(nn.Module):
    def __init__(self):
        super(NegScore,self).__init__()

    def forward(self, node, neg_node):
        # tile node
        th = torch.sum(torch.matmul(node,neg_node.t().contiguous()))
        return th
