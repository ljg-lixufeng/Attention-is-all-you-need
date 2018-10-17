import tensorflow as tf
import numpy as np
from tensorflow import contrib


def position_encode(model_dim, sentence_length, batch_size):
    encoded_vec = np.array(
        [pos / np.power(10000, 2 * i / model_dim)
         for pos in range(sentence_length)
         for i in range(model_dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    single_positon = encoded_vec.reshape([sentence_length, model_dim])
    expand_dim = np.expand_dims(single_positon, axis=0)
    batch_positon = np.tile(single_positon, [batch_size, 1, 1])

    return batch_positon

class Net:
    def __init__(self, is_inference, batch_size, in_vocab_size,
                 tar_vocab_size, embedding_size, lr, ff_dim, dim_k,
                 dim_v, max_seq_len):
        self.batch_size = batch_size
        self.in_vocab_size = in_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.is_inference = is_inference
        self.ff_dim = ff_dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.max_len = max_seq_len

        with tf.name_scope("placeholder"):
            self.input = tf.placeholder(
                shape=[batch_size, max_seq_len], dtype=tf.int32, name="input")
            self.target = tf.placeholder(
                shape=[batch_size, max_seq_len], dtype=tf.int32, name="target")
            self.input_length = tf.placeholder(
                shape=[batch_size], dtype=tf.int32, name="input_length")
            self.target_length = tf.placeholder(
                shape=[batch_size], dtype=tf.int32, name="target_length")
            self.input_position = tf.placeholder(
                shape=[batch_size, max_seq_len, self.embedding_size],
                dtype=tf.float32, name="input_position")
            self.target_position = tf.placeholder(
                shape=[batch_size, max_seq_len, self.embedding_size],
                dtype=tf.float32, name="target_positon")
            self.global_step = tf.Variable(0, trainable=False)


    def positional_encoding(self,):
        in_embed_params, tar_embed_params = self.embedding()
        input_embed = tf.nn.embedding_lookup(
            params=in_embed_params, ids=self.input)
        target_embed = tf.nn.embedding_lookup(
            params=tar_embed_params, ids=self.target)

        input_positional_encoding = tf.sigmoid(
            tf.add(input_embed, self.input_position))
        target_positional_encoding = tf.sigmoid(
            tf.add(target_embed, self.target_position))
        return input_positional_encoding, target_positional_encoding

    # mh_input: 上一层multil head 的input
    # mh_output: 上一层multil head的output
    # layer_norm 不知道能不能用batch norm代替
    def add_norm(self, x, sub_layer_x, num):
        with tf.variable_scope(f"add-and-norm-{num}"):
            # with Residual connection
            return contrib.layers.layer_norm(tf.add(x, sub_layer_x))  

    def feed_forward(self, x, ff_dim, model_dim):
        # out：[batch_size, seq_len, model_dim]
        ff_out = tf.layers.dense(
            inputs=x, units=ff_dim, activation=tf.nn.relu)
        return tf.layers.dense(
            inputs=ff_out, units=model_dim)

    def embedding(self):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            in_embed_params = tf.Variable(
                tf.random_uniform(
                    shape=[self.in_vocab_size, self.embedding_size],
                    minval=-1.0, maxval=1.0),
                name='embedding',
                dtype=tf.float32)
            tar_embed_params = tf.Variable(
                tf.random_uniform(
                    shape=[self.tar_vocab_size, self.embedding_size],
                    minval=-1.0, maxval=1.0))
        return in_embed_params, tar_embed_params


    # linear_num  必须是偶数, 线性层不太一样，最后看效果
    def linear(self, tensor, head_num, linear_dim):
        batch_size, seq_len, embed_size = tensor.shape
        dense_ = tf.layers.dense(tensor, linear_dim, use_bias=False)
        t_shape = dense_.get_shape().as_list()
        reshape0 = tf.reshape(dense_, [batch_size]+[-1]+[head_num, linear_dim // head_num])
        transe0 = tf.transpose(reshape0, [0, 2, 1, 3])
        return transe0

    def scaled_Dot_Product_Attention(self, K, Q, V, is_mask=False):
        batch_size, head_num, seq_len, embedd_size = Q.shape
        qk_mul = tf.matmul(Q, K, transpose_b=True)

        scaled = tf.multiply(
            qk_mul, 1 / tf.sqrt(self.dim_k / head_num.value),
            name='scale')
        if is_mask:
            mask = self.mask(scaled, self.target_length)
        else: mask = scaled
        softmax_out = tf.nn.softmax(mask)
        self.attention_map = softmax_out
        # [batch_size, linear_size, seq_len, emb_size]
        qv_mul = tf.matmul(softmax_out, V) 
        # [batch_size, seq_len, linear_size, emb_sizes]
        qv_trans = tf.transpose(qv_mul, [0, 2, 1, 3])
        # [batch_size, seq_len, -1]
        qv_reshpe = tf.reshape(qv_trans, shape=(batch_size, seq_len, -1))
        return qv_reshpe

    def multi_head_attention(self, Q, K, V, head_num, is_mask=False):
        batch_size, seq_len, embedd_size = Q.shape
        # [batch_size, seq_len, embed_size]

        q_linear = self.linear(Q, head_num, self.dim_k)
        k_linear = self.linear(K, head_num, self.dim_k)
        v_linear = self.linear(V, head_num, self.dim_v)
        # [batch_size, seq_len, tar_vocab_size]
        concated = self.scaled_Dot_Product_Attention(
            q_linear, k_linear, v_linear, is_mask=is_mask)
        # [batch_size, seq_len, embed_size]
        out = tf.layers.dense(concated, units=embedd_size)
        return out

    def mask(self, x, x_length):
        batch_size, head_num, seq_len, emb_dim = x.shape
        mask_0 = tf.sequence_mask(
            lengths=x_length, maxlen=seq_len, dtype=tf.bool)
        mask_1 = tf.reshape(mask_0, shape=[batch_size, 1, seq_len, 1])
        mask = tf.tile(mask_1, multiples=[1, head_num, 1, emb_dim])
        pad = tf.zeros_like(x)
        # 一定要看看这里对不对
        return tf.where(mask, x, pad)

    def encoder(self, x):
        with tf.variable_scope('encode'):
            mh_out = self.multi_head_attention(x, x, x, head_num=8)
            add_norm_1 = self.add_norm(x=x, sub_layer_x=mh_out, num=1)
            feed_forward_out = self.feed_forward(
                add_norm_1, ff_dim=self.ff_dim, model_dim=self.embedding_size)
            add_norm_2 = self.add_norm(
                x=add_norm_1, sub_layer_x=feed_forward_out, num=2)
        return add_norm_2

    def decoder(self, encoder_out, y):
        # #[batch_size, seq_len, model_dim]
        with tf.variable_scope('decode'):
            # #[batch_size, seq_len, model_dim]
            mh1_out = self.multi_head_attention(
                y, y, y, head_num=8, is_mask=True)
            add_norm_1 = self.add_norm(x=y, sub_layer_x=mh1_out, num=1)
            mh2_out = self.multi_head_attention(
                Q=encoder_out, K=encoder_out, V=add_norm_1, head_num=8)
            add_norm_2 = self.add_norm(
                x=add_norm_1, sub_layer_x=mh2_out, num=2)
            ff_out = self.feed_forward(
                x=add_norm_2, ff_dim=self.ff_dim,
                model_dim=self.embedding_size)
            add_norm_3 = self.add_norm(x=add_norm_2,
                                       sub_layer_x=ff_out, num=3)
            # [batch_size, seq_len, model_dim]
            return add_norm_3



