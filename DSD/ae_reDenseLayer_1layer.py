import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder(object):
    def __init__(self, data_len, para_dict):
        self.data_len = data_len

        self.learning_rate = 0.001
        self.batch_size = 64

        self.enc_layer_1 = None
        self.dec_layer_1 = None
        self.enc_output = None
        self.loss = None
        self.optimizer = None
        self.sparsity = 0.7

        self.w_enc = para_dict['w_enc']
        self.b_enc = para_dict['b_enc']
        self.w_dec = para_dict['w_dec']
        self.b_dec = para_dict['b_dec']

    def _init(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_len])

        self.w_enc_h1 = tf.get_variable(name='w_restore_enc_h1', initializer=self.w_enc)
        self.w_dec_h1 = tf.get_variable(name='w_restore_dec_h1', initializer=self.w_dec)
        self.b_enc_h1 = tf.get_variable(name='b_restore_enc_h1', initializer=self.b_enc)
        self.b_dec_h1 = tf.get_variable(name='b_restore_dec_h1', initializer=self.b_dec)

    # TODO: high variance win
    def wta(self, _input, wta_rate=0.9):
        shape = tf.shape(_input)
        b_size, feat_size = shape[0], shape[1]
        k = tf.cast(wta_rate * tf.cast(feat_size, tf.float32), tf.int32)
        th_k, _ = tf.nn.top_k(_input, k)
        winner_min = th_k[:, k - 1: k]  # minimization of winner_hidden
        drop = tf.where(_input < winner_min, tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        _input *= drop
        return _input

    def model(self):
        # Construct model
        self.enc_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.w_enc_h1), self.b_enc_h1))
        self.enc_layer_1 = tf.layers.batch_normalization(inputs=self.enc_layer_1)
        self.dec_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(self.enc_layer_1, self.w_dec_h1), self.b_dec_h1))
        self.dec_layer_1 = tf.layers.batch_normalization(inputs=self.dec_layer_1)

        # Prediction & Targets (Labels) are the input data.
        y_pred, y_true = self.dec_layer_1, self.x

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.square(y_true - y_pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, train_x, test_x, epoch=200, itr=None, display_step=10, ):

        self._init()
        self.model()

        init = tf.global_variables_initializer()
        iteration = len(train_x) // self.batch_size if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(epoch):
                for i in range(iteration):

                    loss, w_enc, b_enc, w_dec, b_dec = \
                        sess.run([self.loss, self.w_enc_h1, self.b_enc_h1, self.w_dec_h1, self.b_dec_h1],
                                 feed_dict={self.x: train_x[i * self.batch_size:(i + 1) * self.batch_size]})

                    print('Iteration %i: Optimization: %f' % (i + 1, loss))
                print('Finished Epoch {0}\n'.format(j))

            train_feat = sess.run(([self.enc_layer_1]), feed_dict={self.x: train_x})
            test_feat = sess.run([self.enc_layer_1], feed_dict={self.x: test_x})
            para_dict = {'w_enc': w_enc, 'b_enc': b_enc, 'w_dec': w_dec, 'b_dec': b_dec}
        return train_feat, test_feat, para_dict
        pass


if __name__ == '__main__':
    pass