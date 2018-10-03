import tensorflow as tf
import numpy as np

class Autoencoder(object):
    def __init__(self, data_len=128):
        self.data_len = data_len
        self.learning_rate = 0.001
        self.batch_size = 64

        self.height = 128
        self.width  = 128

        self.h_2 = 8
        self.w_2 = 8

        self.wta_rate = None

        self.rho = .1
        self.beta = tf.constant(3.)
        self.lam = tf.constant(.001)

    def _init(self):
        # convolution paras
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.height, self.width, 1])
        self.x_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.h_2, self.w_2, 32])

        filter_size1 = 32
        filter_size2 = 64
        kernel_size1 = 2**4
        kernel_size2 = 2**3

        self.f_enc_conv1 = tf.Variable(tf.random_normal([kernel_size1, kernel_size1, 1, filter_size1]))
        self.f_dec_conv1 = tf.Variable(tf.random_normal([kernel_size1, kernel_size1, 1, filter_size1]))

        self.f_enc_conv2 = tf.Variable(tf.random_normal([kernel_size2, kernel_size2, filter_size1, filter_size2]))
        self.f_dec_conv2 = tf.Variable(tf.random_normal([kernel_size2, kernel_size2, filter_size1, filter_size2]))

        self.s_enc_conv1 = [1, kernel_size1, kernel_size1, 1]
        self.s_dec_conv1 = [1, kernel_size1, kernel_size1, 1]

        self.s_enc_conv2 = [1, kernel_size2, kernel_size2, 1]
        self.s_dec_conv2 = [1, kernel_size2, kernel_size2, 1]

        self.output_shape1 = [self.batch_size, self.height, self.width, 1]
        self.output_shape2 = [self.batch_size, int(self.height//kernel_size1), int(self.width//kernel_size1), filter_size1]

    def model_one(self):
        # 128 -> 8
        self.enc_conv1 = tf.nn.relu(tf.nn.conv2d(input=self.x, filter=self.f_enc_conv1,
                                                 strides=self.s_enc_conv1, padding='VALID'))
        self.dec_conv1 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.enc_conv1, filter=self.f_dec_conv1,
                                                              strides=self.s_dec_conv1, output_shape=self.output_shape1))

        self.loss_1_J = tf.reduce_sum(tf.pow(tf.subtract(self.dec_conv1, self.y), 2))
        self.train_op_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_1_J)
        pass

    def model_two(self):
        # 8 -> 1
        self.batch_enc_1 = tf.layers.batch_normalization(inputs=self.x_2)
        self.enc_conv2 = tf.nn.relu(tf.nn.conv2d(input=self.batch_enc_1, filter=self.f_enc_conv2,
                                                 strides=self.s_enc_conv2, padding='VALID'))
        self.dec_conv2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(value=self.enc_conv2, filter=self.f_dec_conv2,
                                                              strides=self.s_dec_conv2, output_shape=self.output_shape2))

        self.loss_2_J = tf.reduce_sum(tf.pow(tf.subtract(self.dec_conv2, self.batch_enc_1), 2))                      # loss
        self.train_op_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_2_J)
        pass

    def KLD(self, p, q):
        invrho = tf.subtract(tf.constant(1.), p)
        invrhohat = tf.subtract(tf.constant(1.), q)
        addrho = tf.add(tf.multiply(p, tf.log(tf.div(p, q))), tf.multiply(invrho, tf.log(tf.div(invrho, invrhohat))))
        return tf.reduce_sum(addrho)

    def wta(self, input_layer):
        shape = tf.shape(input_layer)
        b_size, feat_size = shape[0], shape[1]
        k = tf.cast(self.wta_rate * tf.cast(feat_size, tf.float32), tf.int32)
        th_k, _ = tf.nn.top_k(input_layer, k)
        winner_min = th_k[:, k - 1: k]  # minimization of winner_hidden
        drop = tf.where(input_layer < winner_min, tf.zeros(shape, tf.float32), tf.ones(shape, tf.float32))
        input_layer *= drop
        return input_layer

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

    def train(self, train_x, test_x, SVEB_x=None, VEB_x=None, epoch=200, itr=None, display_step=10, SVEB_and_VEB=False):
        """
        #TODO SVEB & VEB for patient-specific ECG classification
        """

        self._init()
        self.model_one()
        self.model_two()

        init = tf.global_variables_initializer()
        iteration = len(train_x) // self.batch_size if itr is None else itr
        with tf.Session() as sess:
            sess.run(init)
            for j in range(epoch):
                for i in range(iteration):
                    # de-noising and stacked here
                    batch_xs = train_x[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_xs_noise = self.masking_noise(X=batch_xs, v=0.333)
                    _, loss_1_J = sess.run([self.train_op_1, self.loss_1_J],
                                           feed_dict={self.x: batch_xs_noise, self.y: batch_xs})
                    print('sae_loss_1_J', loss_1_J)

            for j in range(epoch):
                for i in range(iteration):
                    batch_xs = train_x[i * self.batch_size:(i + 1) * self.batch_size].astype(np.float32)
                    enc_1 = sess.run([self.enc_conv1], feed_dict={self.x: batch_xs})
                    _, loss_2_J = sess.run([self.train_op_2, self.loss_2_J],
                                           feed_dict={self.x_2: enc_1})
                    print('sae_loss_2_J', loss_2_J)

            enc_1 = sess.run([self.enc_conv1], feed_dict={self.x: train_x})
            train_feature = sess.run([self.enc_conv2], feed_dict={self.x_2: enc_1})
            test_x = sess.run([self.enc_conv1], feed_dict={self.x: test_x})
            test_feature = sess.run([self.enc_conv2], feed_dict={self.x_2: test_x})
            if SVEB_and_VEB is True:
                SVEB_x = sess.run([self.enc_conv1], feed_dict={self.x: SVEB_x})
                SVEB_feature = sess.run([self.enc_conv2], feed_dict={self.x_2: SVEB_x})
                VEB_x = sess.run([self.enc_conv1], feed_dict={self.x: VEB_x})
                VEB_feature = sess.run([self.enc_conv2], feed_dict={self.x_2: VEB_x})
                return train_feature, test_feature, SVEB_feature, VEB_feature
            else:
                return train_feature, test_feature


if __name__ == '__main__':
    pass





