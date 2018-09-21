import os
import numpy as np


def dictionary_from_txt_files(root_dir, files, split_symbol=None):
    print('正在生成字典......, 返回参数：(word2ids, ids2word)')
    w2i_dict = {}
    for file in files:
        file_name = os.path.join(root_dir, file)
        with open(file_name, encoding='utf8') as f:
            fline = f.readline()
            while fline:
                for w in fline.replace('\n', '').split(split_symbol):
                    if w not in w2i_dict:
                        w2i_dict[w] = np.int32(len(w2i_dict))
                fline = f.readline()
    i2w_dict = {i: w for w, i in w2i_dict.items()}
    # print('w2i_dict:', w2i_dict)
    return w2i_dict, i2w_dict
