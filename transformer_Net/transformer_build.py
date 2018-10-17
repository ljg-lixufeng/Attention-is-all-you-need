import tensorflow as tf
from transformer_Net.cnn_attention import Net as attention_base
from tensorflow.contrib import seq2seq
import nltk


class transformer(attention_base):
    def __init__(self, is_inference, batch_size, in_vocab_size,
                 tar_vocab_size, embedding_size, lr, ff_dim, dim_v,
                 dim_k, max_len, rev_target_vocab):
        super(transformer, self).__init__(
            is_inference, batch_size, in_vocab_size,tar_vocab_size,
            embedding_size, lr, ff_dim=ff_dim, dim_k=dim_k, dim_v=dim_v,
            max_seq_len=max_len)
        self.bos = 1
        self.pad = 0
        self.eos = 2
        self.rev_target_vocab = rev_target_vocab

    def _build_encoder(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)
        for i in range(1, 7):
            with tf.variable_scope(f"encoder-layer-{i}"):
                encoder_outputs = self.encoder(o1)
                o1 = tf.identity(encoder_outputs)
        return encoder_outputs

    def _build_decoder(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)
        for i in range(1, 7):
            with tf.variable_scope(f"dcoder-layer-{i}"):
                decoder_outputs = self.decoder(encoder_outputs, o1)
                o1 = tf.identity(decoder_outputs)
        logits = tf.layers.dense(decoder_outputs, self.tar_vocab_size)
        train_predictions = tf.argmax(logits, axis=1, )
        # logits：[batch_size, seq_len, vocabsize]
        return logits, train_predictions

    def _build_loss(self, logits, target, target_length):
        with tf.variable_scope('loss'):
            weight_masks = tf.sequence_mask(lengths = target_length,
                                            maxlen=self.max_len,
                                            dtype=tf.float32)
            loss = seq2seq.sequence_loss(logits, target, weight_masks)
            return loss

    def _build_optimazer(self, loss, lr):
        optimazer = tf.train.AdamOptimizer(learning_rate=lr,)
        train_op = optimazer.minimize(loss=loss, global_step=self.global_step)
        return train_op

    def build_train(self):
        inputs, targets = self.positional_encoding()
        encoder_outputs = self._build_encoder(inputs)
        # [b, s, modle_dim]
        logits, self.train_prediction = self._build_decoder(
            decoder_inputs=targets,
            encoder_outputs=encoder_outputs)
        loss = self._build_loss(
            logits=logits,
            target=self.target,
            target_length=self.target_length)
        train_op = self._build_optimazer(loss=loss, lr=self.lr)
        return train_op, loss

    def build_inference(self):
        inputs, _ = self.positional_encoding()
        encoder_outputs = self._build_encoder(inputs)
        # 1.输入bos, bos=1, eos = 0
        bos_ = tf.ones(shape=[self.batch_size, 1])
        bos = tf.pad(tensor= bos_, paddings=[[0, 0], [0, self.max_len-1]])
        logits, _ = self._build_decoder(
            decoder_inputs=bos, encoder_outputs=encoder_outputs)
        # 2.输入上个step输出的结果
        logits = tf.identity(logits)
        for i in range(self.max_len):
            next_decoder_input = self._next_input(i, logits)
            logits, inference_prediction = self._build_decoder(
                decoder_inputs=next_decoder_input,
                encoder_outputs=encoder_outputs)
            logits = tf.identity(logits)
            predict = tf.identity(inference_prediction)
        return predict


    def _next_input(self, index, decoder_output_logits):
        predict = tf.arg_max(decoder_output_logits, 2)  # [b, s]
        batch_size, seq_len = predict.shape
        next_input_ = tf.slice(input=predict,
                               begin=[0, index],
                               size=[batch_size, 1])
        next_input = tf.pad(input=next_input_,
                            paddings=[[0, 0], [index, seq_len - 1 - index]])
        return next_input # [b,s]

    def _build_metric(self):

        def blue_score(labels, predictions,
                       weights=None, metrics_collections=None,
                       updates_collections=None, name=None):

            def _nltk_blue_score(labels, predictions):

                # slice after <eos>
                predictions = predictions.tolist()
                for i in range(len(predictions)):
                    prediction = predictions[i]
                    if self.eos in prediction:
                        predictions[i] = prediction[:prediction.index(self.eos) + 1]

                rev_target_vocab = self.rev_target_vocab

                labels = [
                    [[rev_target_vocab.get(w_id, "") for w_id in label if w_id != self.pad]]
                    for label in labels.tolist()]
                predictions = [
                    [rev_target_vocab.get(w_id, "") for w_id in prediction]
                    for prediction in predictions]

                print("label: ", labels[0][0])
                print("prediction: ", predictions[0])

                return float(nltk.translate.bleu_score.corpus_bleu(labels, predictions))

            score = tf.py_func(_nltk_blue_score, (labels, predictions), tf.float64)
            return tf.metrics.mean(score * 100.0)

        self.metrics = {
            "bleu": blue_score(self.target, self.train_prediction)
        }

