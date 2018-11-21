import torch
import torch.optim

from AMRNLMinibatch import Minibatch
from AMRNL import AMRNL
from script.Util import loadData, load_config,load_embedding
def train(content, word_embed,len_embed, content_size,user_size, data, config):
    minibatch = Minibatch(data, content_size, config)
    model = AMRNL(content, word_embed,len_embed, user_size,config)
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
                result = 0
                for j in range(minibatch.test_len):
                    question_id, user_id_list, \
                    answer_id_list, answer_vote_list = minibatch.evaluate_next_minibatch()
                    result = result + model.evaluate(question_id, user_id_list, answer_id_list, answer_vote_list)
                print("iteration {}: accuracy {}".format(i, result*1.0/minibatch.test_len))

if __name__ == '__main__':
    config = load_config()
    _, content_len, user_len, content, question_pair = loadData(config.file_dir_list)
    content_embedd, len_embed, word_embed = load_embedding(content)
    train(content_embedd, word_embed, len_embed, content_len, user_len, question_pair, config)