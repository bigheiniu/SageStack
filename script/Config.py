class config:
    batch_size = 20
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
    resource_base_dir='/home/weiying/yichuan/resource/academia.stackexchange.com/'
    ordinary_dir_list = [resource_base_dir + 'Posts.xml', resource_base_dir + 'Votes.xml']
    file_dir_list= [resource_base_dir + 'post.pickle',
                resource_base_dir +'Votes.pickle']
    content_file = resource_base_dir +'content_list.pickle'

    #G, content_len, user_len, content
    store=False
    target_dir_list = ['G.pickle','content.pickle','user_len.pickle']
    target_dir_list = ['/home/weiying/yichuan/resource/academia.stackexchange.com/'+ele for ele in target_dir_list]
    #wordvector feature
    MAX_NB_WORDS = 200000
    EMBEDING_DIM = 300
    EMBEDDING_FILE = resource_base_dir + "GoogleNews-vectors-negative300.bin.gz"
    EDGE_REMOVE_PRO = 0.1
    MAX_SEQUENCE_LENGTH = 30

    debug= True
    on_gpu = True
