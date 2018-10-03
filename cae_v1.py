"""
Author:
https://github.com/c1mone/Tensorflow-101
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(object):
    def __init__(self, data_len=256):
        self.data_len = data_len

        self.data_len = 256
        self.original_height = 1
        self.original_width  = int(self.data_len // self.original_height)
        self.fold_height     = 16
        self.fold_width      = int(self.data_len // self.fold_height)
        self.original_shape = [self.original_height, self.original_width]
        # self.fold_shape = [self.fold_height, self.fold_width]
        self.learning_rate = 0.001
        self.wta_rate = 0.9
        self.batch_size = 32

    def _init(self):
        # convolution paras
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.original_height, self.original_width, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.original_height, self.original_width, 1])

        filter_size = 32
        kernel_size = 2

        self.f_enc_conv1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, filter_size]))
        self.f_dec_conv1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1, filter_size]))

        self.f_enc_conv2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))
        self.f_dec_conv2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))

        self.f_enc_conv3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))
        self.f_dec_conv3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))

        self.f_enc_conv4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))
        self.f_dec_conv4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, filter_size, filter_size]))

        self.s_enc_conv1 = [1, kernel_size, kernel_size, 1]
        self.s_dec_conv1 = [1, kernel_size, kernel_size, 1]

        self.s_enc_conv2 = [1, kernel_size, kernel_size, 1]
        self.s_dec_conv2 = [1, kernel_size, kernel_size, 1]

        self.s_enc_conv3 = [1, kernel_size, kernel_size, 1]
        self.s_dec_conv3 = [1, kernel_size, kernel_size, 1]

        self.s_enc_conv4 = [1, kernel_size, kernel_size, 1]
        self.s_dec_conv4 = [1, kernel_size, kernel_size, 1]

        self.output_shape1 = [self.batch_size, self.fold_height, self.fold_width, 1]
        self.output_shape2 = [self.batch_size, int(self.fold_height//2), int(self.fold_width//2), filter_size]
        self.output_shape3 = [self.batch_size, int(self.fold_height//4), int(self.fold_width//4), filter_size]
        self.output_shape4 = [self.batch_size, int(self.fold_height//8), int(self.fold_width//8), filter_size]

    def encoder(self):
        fold_x = tf.reshape(tensor=self.x, shape=[tf.shape(self.x)[0], self.fold_height, self.fold_width, 1], name='Fold_wave')

        self.enc_conv1 = tf.nn.relu(tf.nn.conv2d(input=fold_x, filter=self.f_enc_conv1,
                                                 strides=self.s_enc_conv1, padding='VALID'))
        # print(self.enc_conv1)
        # self.enc_conv1 = tf.layers.batch_normalization(inputs=self.enc_conv1)
        self.enc_conv2 = tf.nn.relu(tf.nn.conv2d(input=self.enc_conv1, filter=self.f_enc_conv2,
                                                 strides=self.s_enc_conv2, padding='VALID'))
        # print(self.enc_conv2)
        # self.enc_conv2 = tf.layers.batch_normalization(inputs=self.enc_conv2)

        self.enc_conv3 = tf.nn.relu(tf.nn.conv2d(input=self.enc_conv2, filter=self.f_enc_conv3,
                                                 strides=self.s_enc_conv3, padding='VALID'))
        # print(self.enc_conv2)
        # self.enc_conv2 = tf.layers.batch_normalization(inputs=self.enc_conv2)

        self.enc_conv4 = tf.nn.relu(tf.nn.conv2d(input=self.enc_conv3, filter=self.f_enc_conv4,
                                                 strides=self.s_enc_conv4, padding='VALID'))
        # print(self.enc_conv2)
        # self.enc_conv2 = tf.layers.batch_normalization(inputs=self.enc_conv2)
        pass

    def decoder(self):
        self.dec_conv4 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.enc_conv4, filter=self.f_enc_conv4,
                                                              strides=self.s_enc_conv4, output_shape=self.output_shape4))
        # print(self.dec_conv4)

        self.dec_conv3 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.dec_conv4, filter=self.f_enc_conv3,
                                                              strides=self.s_enc_conv3, output_shape=self.output_shape3))
        # print(self.dec_conv3)

        self.dec_conv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.dec_conv3, filter=self.f_enc_conv2,
                                                              strides=self.s_enc_conv2, output_shape=self.output_shape2))
        # print(self.dec_conv2)

        self.dec_conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.dec_conv2, filter=self.f_enc_conv1,
                                                              strides=self.s_enc_conv1, output_shape=self.output_shape1))
        # print(self.dec_conv1)
        self.dec_conv1 = tf.squeeze(self.dec_conv1)
        self.dec_conv1 = tf.reshape(tensor=self.dec_conv1, shape=[tf.shape(self.x)[0], self.original_height, self.original_width])
        self.dec_conv1 = tf.reshape(tensor=self.dec_conv1, shape=[tf.shape(self.x)[0], self.original_height, self.original_width, 1])
        pass

    def model(self):

        self.encoder()
        self.decoder()

        self.loss = tf.reduce_sum(tf.pow((self.dec_conv1 - self.y), 2))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self, train_x, test_x, SVEB_x=None, VEB_x=None, epoch=20, itr=None, SVEB_and_VEB=False):

        # check dimension of inputs
        train_x_noise = self.masking_noise(X=train_x, v=0.333)
        train_x_noise = np.reshape(train_x_noise, newshape=[len(train_x_noise), self.original_height, self.original_width, 1])
        train_x = np.reshape(train_x, newshape=[len(train_x), self.original_height, self.original_width, 1]) if np.ndim(train_x) == 2 else train_x
        test_x = np.reshape(test_x, newshape=[len(test_x), self.original_height, self.original_width, 1]) if np.ndim(test_x) == 2 else test_x

        self._init()
        self.model()

        init = tf.global_variables_initializer()
        iteration = len(train_x) // self.batch_size if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(epoch):
                for i in range(iteration):
                    batch_xs_noise = train_x_noise[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_xs = train_x[i * self.batch_size: (i + 1) * self.batch_size]
                    op, l = sess.run([self.optimizer, self.loss],
                                     feed_dict={self.x: batch_xs_noise, self.y: batch_xs})
                    print('cae_loss:', l)
                print('epoch ', j, ' finished.')
            train_feat = sess.run([self.enc_conv4], feed_dict={self.x: train_x})
            test_feat = sess.run([self.enc_conv4], feed_dict={self.x: test_x})

            if SVEB_and_VEB is True:
                SVEB_x = np.reshape(SVEB_x, newshape=[len(SVEB_x), self.original_height, self.original_width, 1]) if np.ndim(SVEB_x) == 2 else SVEB_x
                VEB_x = np.reshape(VEB_x, newshape=[len(VEB_x), self.original_height, self.original_width, 1]) if np.ndim(VEB_x) == 2 else VEB_x
                SVEB_feat = sess.run([self.enc_conv4], feed_dict={self.x: SVEB_x})
                VEB_feat = sess.run([self.enc_conv4], feed_dict={self.x: VEB_x})
                return train_feat, test_feat, SVEB_feat, VEB_feat
            else:
                return train_feat, test_feat

    def wta(self, _input):
        shape = tf.shape(_input)
        b_size, feat_size = shape[0], shape[1]
        k = tf.cast(self.wta_rate * tf.cast(feat_size, tf.float32), tf.int32)
        th_k, _ = tf.nn.top_k(_input, k)
        winner_min = th_k[:, k - 1: k]  # minimization of winner_hidden
        drop = tf.where(_input < winner_min, tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        _input *= drop
        return _input

    def test_and_show(self, data_set, sess):
        # Test trained model
        batch_test = data_set[:self.batch_size]
        encode_decode = sess.run(self.dec_conv1, feed_dict={self.x: batch_test})
        f, a = plt.subplots(2, 3, figsize=(3, 2))
        for i in range(3):
            a[0][i].plot(np.squeeze(data_set[i]))
            a[1][i].plot(np.squeeze(encode_decode[i]))
            # a[0][i].imshow(np.reshape(data_set[i], [self.height, self.width]))
            # a[1][i].imshow(np.reshape(encode_decode[i], [self.height, self.width]))
        plt.show()
        # correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    def masking_noise(self, X, v):
        """
        :param X: data want to be corrupted
        :param v: corruption ratio
        :return: corrupted data
        """
        X_noise = X.copy()
        n_samples = X.shape[0]
        n_features = X.shape[1]
        v = np.round(v * X.shape[1]).astype(np.int)
        for i in range(n_samples):
            mask = np.random.randint(0, n_features, v)
            for m in mask:
                X_noise[i][m] = 0.
        return X_noise


if __name__ == '__main__':
    # ae = Autoencoder()
    # ae.train()
    pass