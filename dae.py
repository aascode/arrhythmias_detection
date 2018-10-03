import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class Autoencoder(object):
    def __init__(self, data_set):
        self.data_set = data_set

        if len(np.shape(data_set)) == 2:
            (self.data_num, self.data_len) = np.shape(data_set)
        if len(np.shape(data_set)) == 3:
            (self.data_with, self.height) = np.shape(data_set[0])
            pass

        self.learning_rate = 0.001
        self.batch_size = 64

        self.num_hidden_1 = 64
        self.num_hidden_2 = 32
        self.num_hidden_3 = 16
        self.wta_rate = None
        self.plt_l = []

    def _init(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_len])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.data_len])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.data_len, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_3])),
            # 'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_1, self.data_len])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_3, self.num_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([self.num_hidden_1, self.data_len])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.num_hidden_3])),
            # 'decoder_b1': tf.Variable(tf.random_normal([self.data_len])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.data_len])),
        }

    def masking_noise(self, X, v=0.3):
        """
        :param X: data want to be corrupted
        :param v: corruption ratio
        :return: corrupted data
        """
        X_noise = X.copy()
        n_samples = X.shape[0]
        n_features = X.shape[0]
        v = np.round(v * X.shape[1]).astype(np.int)
        for i in range(n_samples):
            mask = np.random.randint(0, n_features, v)
            for m in mask:
                X_noise[i][m] = 0.
        return X_noise

    # Building the encoder
    def encoder(self):
        # Encoder Hidden layer with sigmoid activation #1
        # Encoder Hidden layer with sigmoid activation #2
        # Encoder Hidden layer with sigmoid activation #3
        self.enc_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        self.enc_layer_2 = tf.nn.relu(tf.add(tf.matmul(self.enc_layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        self.enc_layer_3 = tf.nn.relu(tf.add(tf.matmul(self.enc_layer_2, self.weights['encoder_h3']), self.biases['encoder_b3']))

    def wta(self):
        shape = tf.shape(self.enc_layer_3)
        b_size, feat_size = shape[0], shape[1]
        k = tf.cast(self.wta_rate * tf.cast(feat_size, tf.float32), tf.int32)
        th_k, _ = tf.nn.top_k(self.enc_layer_3, k)
        winner_min = th_k[:, k - 1: k]  # minimization of winner_hidden
        drop = tf.where(self.enc_layer_3 < winner_min, tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        self.enc_layer_3 *= drop

    # Building the decoder
    def decoder(self):
        # Decoder Hidden layer with sigmoid activation #1
        # Decoder Hidden layer with sigmoid activation #2
        # Decoder Hidden layer with sigmoid activation #3
        self.dec_layer_1 = tf.nn.relu(tf.add(tf.matmul(self.enc_layer_3, self.weights['decoder_h1']), self.biases['decoder_b1']))
        self.dec_layer_2 = tf.nn.relu(tf.add(tf.matmul(self.dec_layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        self.dec_layer_3 = tf.nn.relu(tf.add(tf.matmul(self.dec_layer_2, self.weights['decoder_h3']), self.biases['decoder_b3']))

    def model(self):
        # Construct model
        # encoder_op = self.encoder(self.x)
        # decoder_op = self.decoder(encoder_op)

        self.encoder()
        if self.wta_rate != None: self.wta()
        self.decoder()

        # Prediction
        y_pred = self.dec_layer_3
        # Targets (Labels) are the input data.
        y_true = self.y

        # Define loss and optimizer, minimize the squared error
        self.loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # self.loss = -tf.reduce_mean(y_true * tf.log(y_pred))
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, train_x, test_x, epoch=20, itr=None, display_step=10):

        self._init()
        self.model()

        init = tf.global_variables_initializer()
        iteration = len(train_x) // self.batch_size if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(epoch):
                for i in range(iteration):
                    train_x_noise = self.masking_noise(train_x[i * self.batch_size:(i + 1) * self.batch_size], 0.5)
                    loss = sess.run([self.loss], feed_dict={
                        self.x: train_x_noise,
                        self.y: train_x[i * self.batch_size:(i + 1) * self.batch_size]})
                    self.plt_l = np.concatenate((self.plt_l, loss))
                    # if i % display_step == 0 or i == 1:
                    #     print('Iteration %i: Minibatch Loss: %f' % (i + 1, loss))
                    # print('Iteration %i: Minibatch Loss: %f' % (i + 1, loss))
                    print('Iteration %i: Optimization: %f' % (i + 1, loss[0]))
                print('Finished Epoch {0}\n'.format(j))
            print('Autoencoder Training Finished \n')
            train_feature = sess.run([self.enc_layer_3], feed_dict={self.x: train_x})
            test_feature = sess.run([self.enc_layer_3], feed_dict={self.x: test_x})
        return train_feature, test_feature


if __name__ == '__main__':
    pass