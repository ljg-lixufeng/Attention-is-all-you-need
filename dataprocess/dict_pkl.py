import pickle


def dictionary_2_pkl_file(dictionary, filename,):
    print('Write dictionary to %s' % filename)
    with open(filename, 'wb') as w:
        pickle.dump(dictionary, w)


def dictionary_from_pkl_file(filename):
    print('从%s读取字典 ......,返回参数：(word2ids, ids2word)' % filename)
    with open(filename, "rb") as f:
        d = pickle.load(f)
    return d