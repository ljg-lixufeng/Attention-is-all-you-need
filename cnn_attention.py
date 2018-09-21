import tensorflow as tf
import numpy as np
from tensorflow import contrib


class Net:
    def __init__(self, is_inference, batch_size, in_vocab_size, tar_vocab_size,
                 embedding_size, hidden_size, lr, ff_dim):
        self.batch_size = batch_size
        self.in_vocab_size = in_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.is_inference = is_inference
        self.ff_dim = ff_dim

        with tf.name_scope("placeholder"):
            self.input = tf.placeholder(
                shape=[batch_size, None], dtype=tf.int32, name="input")
            self.target = tf.placeholder(
                shape=[batch_size, None], dtype=tf.int32, name="target")
            self.input_length = tf.placeholder(
                shape=[batch_size], dtype=tf.int32, name="input_length")
            self.target_length = tf.placeholder(
                shape=[batch_size], dtype=tf.int32, name="target_length")
            self.global_step = tf.Variable(0, trainable=False)

    def positional_encoding(self):
        in_embed_params, tar_embed_params = self.embedding()
        input_embed = tf.nn.embedding_lookup(
            params=in_embed_params, ids=self.input
        )
        target_embed = tf.nn.embedding_lookup(
            params=tar_embed_params, ids=self.target
        )
        input_position_embed = self.position_encoding(
            d=self.embedding_size, sentence_length=self.input_length
        )
        target_position_embed = self.position_encoding(
            d=self.embedding_size, sentence_length=self.target_length
        )
        input_positional_encoding = tf.sigmoid(
            tf.add(input_embed, input_position_embed)
        )
        target_positional_encoding = tf.sigmoid(
            tf.add(target_embed, target_position_embed)
        )
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
            inputs=x, units=ff_dim, activation=tf.nn.relu
        )
        return tf.layers.dense(
            inputs=ff_out, units=model_dim
        )

    def embedding(self):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            in_embed_params = tf.Variable(
                tf.random_uniform(
                    shape=[self.in_vocab_size, self.embedding_size],
                    minval=-1.0, maxval=1.0),
                name='embedding',
                dtype=tf.float32
            )
            tar_embed_params = tf.Variable(
                tf.random_uniform(
                    shape=[self.tar_vocab_size, self.embedding_size],
                    minval=-1.0, maxval=1.0
                )
            )
        return in_embed_params, tar_embed_params

    def position_encoding(self, d, sentence_length, dtype=tf.float32):
        encoded_vec = np.array(
            [pos / np.power(10000, 2 * i / d)
             for pos in range(sentence_length) 
             for i in range(d)]
        )
        encoded_vec[::2] = np.sin(encoded_vec[::2])
        encoded_vec[1::2] = np.cos(encoded_vec[1::2])
        return tf.convert_to_tensor(
            encoded_vec.reshape([sentence_length, d]), dtype=dtype
        )

    # linear_num  必须是偶数, 线性层不太一样，最后看效果
    def linear(self, tensor, head_num, name):
        batch_size, seq_len, embed_size = tf.shape(tensor)
        temp = tf.reshape(tensor, [batch_size, 1, seq_len, embed_size])
        tile_ = tf.tile(input=temp, multiples=[1, head_num, 1, 1])
        # 记得看看tile的shape是不是 batch_size, head_num, seq_len, embed_size

        linear_var = tf.Variable(
            tf.random_uniform(
                shape=[batch_size, head_num, seq_len, embed_size]), 
                name=name)
        # [batch_size, linear_num, seq_len, embed_size]
        linear_mul = tf.multiply(linear_var, tile_) 
        return linear_mul

    def scaled_Dot_Product_Attention(self, K, Q, V, is_mask=False):
        batch_size, linear_num, seq_len, embedd_size = tf.shape(Q)
        qk_mul = tf.matmul(Q, K, transpose_b=True)
        scaled = tf.multiply(
            qk_mul,
            1 / tf.sqrt(tf.cast(seq_len, dtype=tf.float32)),
            name='scale'
        )
        if is_mask:
            mask = self.mask(x=scaled, x_length=self.target_length)
        else: mask = scaled
        softmax_out = tf.nn.softmax(mask)
        # [batch_size, linear_size, seq_len, emb_size]
        qv_mul = tf.matmul(softmax_out, V) 
        # [batch_size, seq_len, linear_size, emb_sizes]
        qv_trans = tf.transpose(qv_mul, [0, 2, 1, 3])
        # [batch_size, seq_len, -1]
        qv_reshpe = tf.reshape(qv_trans, shape=(batch_size, seq_len, -1))

        return qv_reshpe

    def multi_head_attention(self, Q, K, V, head_num, is_mask=False):
        batch_size, linear_num, seq_len, embedd_size = tf.shape(Q)
        # [batch_size, linear_num, seq_len, embed_size]
        q_linear = self.linear(Q, head_num, 'q_linear')
        k_linear = self.linear(K, head_num, 'k_linear')
        v_linear = self.linear(V, head_num, 'v_linear')
        # [batch_size, seq_len, tar_vocab_size]
        concated = self.scaled_Dot_Product_Attention(
            q_linear, k_linear, v_linear, is_mask=is_mask
        )
        # [batch_size, seq_len, embed_size]
        out = tf.layers.dense(concated, units=embedd_size)
        self.attention_map = out
        return out

    def mask(self, x, x_length):
        batch_size, head_num, seq_len, emb_dim = tf.shape(x)
        mask_0 = tf.sequence_mask(
            lengths=x_length, maxlen=seq_len, dtype=tf.bool)
        mask_1 = tf.reshape(mask_0, shape=[batch_size, 1, seq_len, 1])
        mask = tf.tile(mask_1, multiples=[1, head_num, 1, emb_dim])
        # 一定要看看这里对不对
        return tf.where(mask, x)

    def encoder(self, x):
        with tf.VariableScope(
                name='encode', reuse=False, name_scope='encode'):
            mh_out = self.multi_head_attention(x, x, x, head_num=8)
            add_norm_1 = self.add_norm(x=x, sub_layer_x=mh_out, num=1)
            feed_forward_out = self.feed_forward(
                add_norm_1, ff_dim=2048, model_dim=512
            )
            add_norm_2 = self.add_norm(
                x=add_norm_1, sub_layer_x=feed_forward_out, num=2
            )
        return add_norm_2

    def decoder(self, encoder_out, y):
        # #[batch_size, seq_len, model_dim]
        with tf.VariableScope(
                name='decode', reuse=False, name_scope='decoder'):
            # #[batch_size, seq_len, model_dim]
            mh1_out = self.multi_head_attention(
                y, y, y, head_num=8, is_mask=True)
            add_norm_1 = self.add_norm(x=y, sub_layer_x=mh1_out, num=1)
            mh2_out = self.multi_head_attention(
                Q=encoder_out, K=encoder_out, V=add_norm_1, head_num=8
            )
            add_norm_2 = self.add_norm(
                x=add_norm_1, sub_layer_x=mh2_out, num=2)
            ff_out = self.feed_forward(
                x=add_norm_2, ff_dim=self.ff_dim,
                model_dim=self.embedding_size
            )
            add_norm_3 = self.add_norm(x=add_norm_2,
                                       sub_layer_x=ff_out, num=3
                                       )
            # [batch_size, seq_len, model_dim]
            return add_norm_3