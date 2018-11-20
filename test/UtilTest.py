from __future__ import print_function
from script.Util import *

config = load_config()


def test_loadg_raph():
    list_dir = ['/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/post.pickle','/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/Votes.pickle']
    G, questionLen, questionContent, answerContent  = loadData(list_dir)
    print(G.edges(data=True))
    print(G.nodes(data=True))
    print(questionLen)
    print(questionContent[1])
    print(answerContent[1])
    print(G[0])

def test_embedd():

    list_dir = ['/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/post.pickle','/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/Votes.pickle']
    _, _, quesitonContent, answerContent = loadData(list_dir)
    th = load_embedding(quesitonContent, answerContent)
    for embed in th:
        print(embed.size())



if __name__ == '__main__':
    test_loadg_raph()