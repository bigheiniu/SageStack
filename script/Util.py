from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import pandas as pd
import torch
from torch.autograd import Variable
import numpy as np
import pickle
import random
import networkx as nx
from .TestPreprocess import text_to_wordlist
from .Config import config
# TODO: 配置文件需要静态类去表示　＝＞　pytorch　是否有这种帮助
MAX_NB_WORDS = config.MAX_NB_WORDS
EMBEDING_DIM = config.EMBEDING_DIM
EMBEDDING_FILE = config.EMBEDDING_FILE
EDGE_REMOVE_PRO = config.EDGE_REMOVE_PRO
MAX_SEQUENCE_LENGTH = config.MAX_SEQUENCE_LENGTH




def tensor2numpy_int(x):
    if config.on_gpu:
        return x.cpu().numpy()
    else:
        return x.numpy()
def numpy2tensor_float(x,need_grad=False):
    result = Variable(torch.tensor(x), requires_grad=need_grad)
    if config.on_gpu:
        result = result.type(torch.cuda.FloatTensor)
    else:
        result = result.type(torch.FloatTensor)
    return result


def numpy2tensor_long(x,need_grad=False):
    result = Variable(torch.tensor(x), requires_grad=need_grad)
    if config.on_gpu:
        result = result.type(torch.cuda.LongTensor)
    else:
        result = result.type(torch.LongTensor)
    return result
def numpy2tensor_int(x,need_grad=False):
    result = Variable(torch.tensor(x), requires_grad=need_grad)
    if config.on_gpu:
        result = result.type(torch.cuda.IntTensor)
    else:
        result = result.type(torch.IntTensor)
    return result

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def load_embedding(content):
    #TODO: not use pretrained word2vec
    # input content 是文字内容
    content_len_embed = np.array([len(line) if(len(line) <= MAX_SEQUENCE_LENGTH) else MAX_SEQUENCE_LENGTH for line in content])
    # kears convert wordinto vector
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(content)
    contentId= tokenizer.texts_to_sequences(content)
    # answerContentId = tokenizer.texts_to_sequences(answer_content)
    word_index = tokenizer.word_index
    content_embed = pad_sequences(contentId, maxlen=MAX_SEQUENCE_LENGTH)
    # answerContentEmbed = pad_sequences(answerContentId, maxlen=MAX_SEQUENCE_LENGTH)
    nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
    th = np.max(list(word_index.values()))
    word_embed = np.zeros((nb_words, EMBEDING_DIM))


    #load word2vector vector
    if (config.debug == False):
        word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
        for word, fu in word_index.items():
            if word in word2vec.vocab:
                word_embed[fu] = word2vec.word_vec(word)
    import gc
    gc.enable()
    del content
    gc.collect()
    return content_embed,content_len_embed, word_embed


def load_config():
    from .Config import config
    return config


def loadData(file_dir_list):
    # pandas => networkx graph
    # some problem will not be answered by others
    post_dir = file_dir_list[0]
    vote_dir = file_dir_list[1]
    with open(post_dir, 'rb') as f1:
        post = pickle.load(f1)
    with open(vote_dir, 'rb') as f1:
        vote = pickle.load(f1)
    # vote.drop(['UserId'], inplace=True, axis=1)
    vote = vote[vote['VoteTypeId'] == '2'].groupby('PostId').size().reset_index(name='counts')

    # questionId - userId - answerId
    with open(config.content_file,'rb') as f1:
        content = pickle.load(f1)

    content = text_to_wordlist(content)

    # dense
    content_len = len(content)
    content_id2idx = {value:index for index, value in enumerate(post['Id'].values)}

    question = post[post['PostTypeId'] == '1'][['Id']]
    question_len = len(question)
    answer = post[post['PostTypeId'] == '2'][['Id','ParentId','OwnerUserId']]
    question_answer = pd.merge(question, answer, how='inner',left_on='Id',right_on='ParentId')
    question_answer = question_answer[['Id_x', 'Id_y', 'OwnerUserId']]
    #remove quesiton only with one answer
    answer_count = question_answer.groupby('Id_x')['Id_y'].count()
    answer_count = answer_count[answer_count > 1]
    answer_count = pd.DataFrame(answer_count)
    answer_count['Id'] = answer_count.index
    question_answer = question_answer.merge(answer_count,how='inner',left_on='Id_x',right_on='Id')
    question_answer.drop(['Id_y_y','Id'],inplace=True, axis=1)
    question_answer.columns = ['Id_x', 'Id_y', 'OwnerUserId']
    quesiton_answer_vote = pd.merge(question_answer, vote, how='left', left_on='Id_y', right_on='PostId')
    quesiton_answer_vote.fillna(0., inplace=True)
    quesiton_answer_vote.drop(['PostId'],inplace=True, axis=1)
    print(quesiton_answer_vote.head())
    quesiton_answer_vote.columns = ['q_id', 'a_id', 'u_id', 'score']

    # dense userId
    # remove user who does not answer question
    quesiton_answer_vote['u_id'] = quesiton_answer_vote['u_id'].astype(int)
    userId = quesiton_answer_vote['u_id'].values
    userId = np.unique(userId)
    userRank = list(range(len(userId)))
    userId2idx = {key: value for key, value in zip(userId, userRank)}
    user_len = len(userId)
    #convert userId, answerId, userId
    quesiton_answer_vote['q_id'] = quesiton_answer_vote['q_id'].apply(lambda x: content_id2idx.get(x))
    quesiton_answer_vote['u_id'] = quesiton_answer_vote['u_id'].apply(lambda x: userId2idx.get(x) + content_len)
    quesiton_answer_vote['a_id'] = quesiton_answer_vote['a_id'].apply(lambda x: content_id2idx.get(x))

    quesiton_answer_vote.groupby('q_id').agg({'score': 'sum'})
    score_sum = quesiton_answer_vote.groupby('q_id').agg({'score': 'sum'})
    middle = pd.merge(quesiton_answer_vote,score_sum,how='inner',on='q_id')
    quesiton_answer_vote['score'] = middle['score_x'] / middle['score_y']
    quesiton_answer_vote.fillna(0.001, inplace=True)
    G = nx.from_pandas_edgelist(quesiton_answer_vote, 'q_id', 'u_id', ['a_id', 'score'])
    #build bipartite graph
    nodes = G.nodes()
    attr = dict(nodes)
    for node in attr.keys():
        if int(node) > content_len:
            attr.update({node: {'bipartite': 1}})
        else:
            attr.update({node: {'bipartite': 0}})
    nx.set_node_attributes(G, attr)
    #build validation and training data
    attr = dict()
    value = 0.1
    for edge in list(G.edges()):
        th = random.random()
        if (th < value):
            attr[edge] = True
        else:
            attr[edge] = False
    nx.set_edge_attributes(G, attr, 'train_removed')
    return G, content_len, user_len, content

def saveLoadData(**kwargs):
    value = list(kwargs.values())
    for i,file in enumerate(config.target_dir_list):
        with open(file,'wb') as f1:
            pickle.dump(value[i], f1)
def loadSaveData():
    th = []
    for file in config.target_dir_list:
        with open(file,'rb') as f1:
            data = pickle.load(f1)
        th.append(data)
    with open(config.content_file,'rb') as f1:
        data = pickle.load(f1)
    th.append(data)
    return tuple(th)