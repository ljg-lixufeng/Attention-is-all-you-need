import tensorflow as tf
from cnn_attention import Net as attention_base
from tensorflow.contrib import seq2seq


class transformer(attention_base):
    def __init__(self, is_inference, batch_size, in_vocab_size,
                 tar_vocab_size, embedding_size,
                 hidden_size, lr, ff_dim):
        super(transformer, self).__init__(
            is_inference, batch_size, in_vocab_size,
            tar_vocab_size,embedding_size, hidden_size, lr, ff_dim)

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
        prediction = tf.nn.softmax(logits)
        # [batch_size, seq_len, model_dim]
        return logits, prediction

    def _build_loss(self, logits, target, target_length):
        with tf.variable_scope('loss'):
            weight_masks = tf.sequence_mask(
                lengths = target_length,
                maxlen=tf.reduce_max(target_length)
            )
            loss = seq2seq.sequence_loss(logits, target, weight_masks)
            return loss

    def _build_optimazer(self, loss, lr):
        optimazer = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = optimazer.minimize(loss=loss)
        return train_op

    def build_train(self):
        inputs, targets = self.positional_encoding()
        encoder_outputs = self._build_encoder(inputs)
        # [b, s, modle_dim]
        logits, self.prediction = self._build_decoder(
            decoder_inputs=targets,
            encoder_outputs=encoder_outputs
        )
        loss = self._build_loss(
            logits=logits, target=targets, target_length=self.target_length
        )
        train_op = self._build_optimazer(loss=loss, lr=self.lr)
        return train_op

    def build_inference(self):
        pass

