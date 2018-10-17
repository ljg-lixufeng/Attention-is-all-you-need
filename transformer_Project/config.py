class configer:
    root_dir = r'C:\Users\Administrator\AnacondaProjects\attention_is_all_you_need'

    sentence_file = 'europarl-v7.de-en.en'
    label_file = 'europarl-v7.de-en.de'

    batch_size = 2 # 32
    epoch = 1 # 100
    embedding_size = 50 # 512
    lr = 0.001
    ff_dim = 128 #512
    max_seq_len = 1000
    dim_k = 80 # 512
    dim_v = 80 #512

    test_ssh = 000

