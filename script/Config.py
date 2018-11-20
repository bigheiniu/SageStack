class config:
    batch_size = 10
    lstm_input_size = 300
    lstm_hidden_size = 300
    neg_sample_size = 10
    drop_out = 0.3
    samples_1 = 2
    samples_2 = 4
    dim_1 = 300
    dim_2 = 300
    edge_sample = 100000
    max_degree = 10
    ordinary_dir_list = ['/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/resource/apple/Posts.xml', '/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/resource/apple/Votes.xml']
    file_dir_list= ['/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/post.pickle',
                '/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/Votes.pickle']
    #wordvector feature
    MAX_NB_WORDS = 200000
    EMBEDING_DIM = 300
    EMBEDDING_FILE = "/home/bigheiniu/course/ASU_Course/472/coursePro/472Project/resource/GoogleNews-vectors-negative300.bin.gz"
    EDGE_REMOVE_PRO = 0.1
    MAX_SEQUENCE_LENGTH = 30

    debug= False
    on_gpu = False
