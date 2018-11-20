from script.minibatch import *
from script.Util import loadData

def testminibatch():
    list_dir = ['/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/post.pickle',
                '/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/Votes.pickle']
    G,value,_,_ = loadData(list_dir)
    minbatch = Minibatch(G, 10, 2,value)

    # test adjancy matrix
    th = minbatch.construct_adj()
    for ele in th:
        print(len(ele))
    #test
    print(minbatch.val_mini_batch())

if __name__ == '__main__':
    testminibatch()