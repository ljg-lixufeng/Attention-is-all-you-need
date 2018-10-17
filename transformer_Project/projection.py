import tensorflow as tf
from transformer_Project.data_input import getBatch
from transformer_Project.config import configer as conf
from transformer_Net.transformer_build import transformer
from transformer_Net.cnn_attention import position_encode
from transformer_Project.log_func import my_log

def train_projection():
    logger = my_log('attention')
    data = getBatch()
    next_element = data.get_batch(
        batch_size=conf.batch_size, epoch=conf.epoch)
    net = transformer(is_inference=False,
                      batch_size=conf.batch_size,
                      in_vocab_size=len(data.en_w2i),
                      tar_vocab_size=len(data.de_w2i),
                      embedding_size=conf.embedding_size,
                      lr=conf.lr,
                      ff_dim=conf.ff_dim,
                      dim_k=conf.dim_k,
                      dim_v=conf.dim_v,
                      max_len=conf.max_seq_len,
                      rev_target_vocab=data.de_i2w
                      )
    train_op, loss = net.build_train()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=3, max_to_keep=3)
    with tf.Session() as sess:
        saver.restore(sess, 'log/attention-139400')
        while True:
            batch_input, input_length, batch_target, target_length = sess.run(next_element)
            input_position = position_encode(conf.embedding_size, conf.max_seq_len, conf.batch_size)
            target_position = position_encode(conf.embedding_size, conf.max_seq_len, conf.batch_size)
            feed = {net.input:batch_input, net.target:batch_target,
                    net.input_length:input_length, net.target_length:target_length,
                    net.input_position:input_position, net.target_position:target_position}
            _, loss_var, prediction, global_s = sess.run(
                [train_op, loss, net.train_prediction, net.global_step],
                feed_dict=feed)
            if global_s%100 == 0:
                saver.save(sess, 'log/attention', global_step=global_s)
            logger.info('loss:%s step: %s'%(loss_var, global_s))
            #print('loss',loss_var)
            #print(prediction[0])

def inference_projection():
    logger = my_log('inference attention')
    data = getBatch()
    next_element = data.get_batch(
        batch_size=conf.batch_size, epoch=conf.epoch)
    net = transformer(is_inference=True,
                      batch_size=conf.batch_size,
                      in_vocab_size=len(data.en_w2i),
                      tar_vocab_size=len(data.de_w2i),
                      embedding_size=conf.embedding_size,
                      lr=conf.lr,
                      ff_dim=conf.ff_dim,
                      dim_k=conf.dim_k,
                      dim_v=conf.dim_v,
                      max_len=conf.max_seq_len,
                      rev_target_vocab=data.de_i2w
                      )
    inference_predict = net.build_inference()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'log/attention-139400')
        while True:
            (batch_input, input_length,
             batch_target, target_length) = sess.run(next_element)
            input_position = position_encode(conf.embedding_size, conf.max_seq_len, conf.batch_size)
            target_position = position_encode(conf.embedding_size, conf.max_seq_len, conf.batch_size)
            feed = {net.input: batch_input, net.target: batch_target,
                    net.input_length: input_length, net.target_length: target_length,
                    net.input_position: input_position, net.target_position: target_position}
            predict = sess.run([inference_predict], feed_dict=feed)

