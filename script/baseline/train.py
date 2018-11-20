import torch
import torch.optim

from .AMRNLMinibatch import Minibatch
from .AMRNL import AMRNL
from script.Util import loadData
def train(content, word_embed, user_size, content_size, data, config):
    minibatch = Minibatch(data, content_size, config)
    model = AMRNL(content, word_embed, user_size)
    paramer = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(paramer, lr=0.001)
    # print("optimizer {}".format(optimizer))
    i = 0

    device = torch.device('cuda') if config.on_gpu else torch.device('cpu')
    model.to(device)
    while i < config.edge_sample:
        # time1 = time.time()
        question_id,user_id_list, \
        answer_id_list, answer_vote_list  = minibatch.next_minibatch()
        loss = model.forward(question_id, user_id_list, answer_id_list, answer_vote_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i = i+1
        if i % 100 == 0:
            with torch.no_grad():
                model = model.eval()
            result = 0
            for j in range(minibatch.test_len):
                question_id, user_id_list, \
                answer_id_list, answer_vote_list = minibatch.evaluate_next_minibatch()
                result = result + model.eval(question_id, user_id_list, answer_id_list, answer_vote_list)
            print("iteration {}: accuracy {}".format(i, result*1.0/minibatch.test_len))

if __name__ == '__main__':
    