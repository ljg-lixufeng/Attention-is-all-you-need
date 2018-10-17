import pickle
import numpy as np
import tensorflow as tf
import os
from transformer_Project.config import configer as conf

class getBatch():
    def __init__(self):

        self.en_w2i = pickle.load(
            open(os.path.join(conf.root_dir, 'data', 'en_w2i_map'), 'rb'),
            encoding='iso-8859-1')
        self.en_i2w = pickle.load(
            open(os.path.join(conf.root_dir, 'data', 'en_i2w_map'), 'rb'),
            encoding='iso-8859-1')
        self.de_w2i = pickle.load(
            open(os.path.join(conf.root_dir, 'data', 'de_w2i_map'), 'rb'),
            encoding='iso-8859-1')
        self.de_i2w = pickle.load(
            open(os.path.join(conf.root_dir, 'data', 'de_i2w_map'), 'rb'),
            encoding='iso-8859-1')

    def get_batch(self, batch_size, epoch):
        s_dict = self.en_w2i
        l_dict = self.de_w2i

        def map_func_data(s, l):
            # function：return (batch data, true batch length)
            # with stype (array, array)
            # input: sentences of string
            def my_func(sentences, labels):

                sentences_2id = np.asarray(
                    [[s_dict[word] for word in line.decode('utf8').split()] for line in sentences]
                )
                labels_2id = np.asarray(
                    [[l_dict[word] for word in line.decode('utf8').split()] for line in labels]
                )

                sentences_length = [len(a) for a in sentences_2id]
                labels_length = [len(a) for a in labels_2id]
                #  max length of batch data
                # max_sentence_length = max(sentences_length)
                # max_label_length = max(labels_length)
                max_sentence_length = conf.max_seq_len
                max_label_length = conf.max_seq_len
                # padding
                sentences_padded = [
                    np.pad(d,
                           ((0, max_sentence_length - len(d))),
                           'constant', constant_values=((0, 0))
                           )
                    for d in sentences_2id
                ]
                labels_padded = [
                    np.pad(d,
                           ((0, max_label_length - len(d))),
                           'constant', constant_values=((0,0))
                           ) for d in labels_2id
                ]
                sentences_padded = sentences_padded[:100]
                labels_padded = labels_padded[:100]
                return np.int32(sentences_padded), \
                       np.int32(sentences_length),\
                       np.int32(labels_padded), \
                       np.int32(labels_length)

            sentences_padded_, sentences_length_, labels_padded_, labels_length_ = tf.py_func(
                my_func, [s, l], [tf.int32, tf.int32, tf.int32, tf.int32]
            )
            return sentences_padded_, sentences_length_, labels_padded_, labels_length_

        sentences_data = tf.data.TextLineDataset(
            os.path.join(conf.root_dir, 'data', conf.sentence_file))
        labels_data = tf.data.TextLineDataset(
            os.path.join(conf.root_dir, 'data', conf.label_file))
        data = tf.data.Dataset.zip((sentences_data, labels_data))
        data = data.batch(batch_size=batch_size)
        data = data.repeat(count=100)
        data = data.map(map_func_data, num_parallel_calls=4)
        iterator = data.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element

# 测试
# with tf.Session() as sess:
#     print(sess.run(getBatch().get_batch(1,1)))