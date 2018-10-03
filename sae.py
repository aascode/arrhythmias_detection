import tensorflow as tf
import numpy as np

class Autoencoder(object):
    def __init__(self, data_len=256):
        self.data_len = data_len
        self.learning_rate = 0.001
        self.batch_size = 128

        self.num_hidden_1 = 128
        self.num_hidden_2 = 32
        self.wta_rate = None

        self.rho = .1
        self.beta = tf.constant(3.)
        self.lam = tf.constant(.001)
        self.w_model_one_init = np.sqrt(6. / (self.data_len + self.num_hidden_1))

    def _init(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.data_len])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.data_len])
        self.hidden_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.num_hidden_1])
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_uniform(shape=[self.data_len, self.num_hidden_1], minval=-self.w_model_one_init, maxval=self.w_model_one_init)),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_1, self.data_len])),

            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1]))}
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([self.data_len])),

            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.num_hidden_1]))}

    def model_one(self):
        self.encoder_1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights["encoder_h1"]), self.biases["encoder_b1"]))
        self.decoder_1 = tf.nn.relu(tf.add(tf.matmul(self.encoder_1, self.weights["decoder_h1"]), self.biases["decoder_b1"]))

        self.loss_1_J = tf.reduce_sum(tf.pow(tf.subtract(self.decoder_1, self.y), 2))                                   # loss
        loss_1_rho_hat = tf.div(tf.reduce_sum(self.encoder_1), self.num_hidden_1)                                       # cost sparse
        loss_1_sparse = tf.multiply(self.beta, self.KLD(self.rho, loss_1_rho_hat))                                      # cost sparse
        loss_1_reg = tf.multiply(self.lam, tf.add(tf.nn.l2_loss(self.weights["encoder_h1"]),
                                                     tf.nn.l2_loss(self.weights["decoder_h1"])))                        # cost reg
        self.loss_1_sum = tf.add(tf.add(self.loss_1_J, loss_1_reg), loss_1_sparse)
        self.train_op_1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_1_sum)

    def model_two(self):
        self.batch_hidden_1 = tf.layers.batch_normalization(inputs=self.hidden_1)
        self.encoder_2 = tf.nn.relu(tf.add(tf.matmul(self.batch_hidden_1, self.weights["encoder_h2"]), self.biases["encoder_b2"]))
        self.decoder_2 = tf.nn.relu(tf.add(tf.matmul(self.encoder_2, self.weights["decoder_h2"]), self.biases["decoder_b2"]))

        self.loss_2_J = tf.reduce_sum(tf.pow(tf.subtract(self.decoder_2, self.batch_hidden_1), 2))                      # loss
        loss_2_rho_hat = tf.div(tf.reduce_sum(self.encoder_2), self.num_hidden_2)                                       # cost sparse
        loss_2_sparse = tf.multiply(self.beta, self.KLD(self.rho, loss_2_rho_hat))                                      # cost sparse
        loss_2_reg = tf.multiply(self.lam, tf.add(tf.nn.l2_loss(self.weights["encoder_h2"]),
                                                     tf.nn.l2_loss(self.weights["decoder_h2"])))                        # cost reg
        self.loss_2_sum = tf.add(tf.add(self.loss_2_J, loss_2_reg), loss_2_sparse)
        self.train_op_2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_2_sum)

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

    def train(self, train_x, test_x, SVEB_x=None, VEB_x=None, epoch=200, itr=None, display_step=10, SVEB_and_VEB=False, corrupt=False):
        """
        #TODO SVEB & VEB for patient-specific ECG classification
        """
        if corrupt is False:
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
                        batch_xs = sess.run([self.encoder_1], feed_dict={self.x: batch_xs})
                        _, loss_2_J = sess.run([self.train_op_2, self.loss_2_J],
                                               feed_dict={self.hidden_1: np.squeeze(batch_xs)})
                        print('sae_loss_2_J', loss_2_J)

                train_x = sess.run([self.encoder_1], feed_dict={self.x: train_x})
                train_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(train_x)})
                test_x = sess.run([self.encoder_1], feed_dict={self.x: test_x})
                test_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(test_x)})
                if SVEB_and_VEB is True:
                    SVEB_x = sess.run([self.encoder_1], feed_dict={self.x: SVEB_x})
                    SVEB_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(SVEB_x)})
                    VEB_x = sess.run([self.encoder_1], feed_dict={self.x: VEB_x})
                    VEB_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(VEB_x)})
                    return train_feature, test_feature, SVEB_feature, VEB_feature
                else:
                    return train_feature, test_feature

        elif corrput is True:
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
                        batch_xs_noise = self.masking_noise(X=batch_xs, v=0.777)
                        _, loss_1_J = sess.run([self.train_op_1, self.loss_1_J],
                                               feed_dict={self.x: batch_xs, self.y: batch_xs_noise})
                        print('sae_loss_1_J', loss_1_J)

                for j in range(epoch):
                    for i in range(iteration):
                        batch_xs = train_x[i * self.batch_size:(i + 1) * self.batch_size].astype(np.float32)
                        batch_xs = sess.run([self.encoder_1], feed_dict={self.x: batch_xs})
                        _, loss_2_J = sess.run([self.train_op_2, self.loss_2_J],
                                               feed_dict={self.hidden_1: np.squeeze(batch_xs)})
                        print('sae_loss_2_J', loss_2_J)

                train_x = sess.run([self.encoder_1], feed_dict={self.x: train_x})
                train_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(train_x)})
                test_x = sess.run([self.encoder_1], feed_dict={self.x: test_x})
                test_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(test_x)})
                if SVEB_and_VEB is True:
                    SVEB_x = sess.run([self.encoder_1], feed_dict={self.x: SVEB_x})
                    SVEB_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(SVEB_x)})
                    VEB_x = sess.run([self.encoder_1], feed_dict={self.x: VEB_x})
                    VEB_feature = sess.run([self.encoder_2], feed_dict={self.hidden_1: np.squeeze(VEB_x)})
                    return train_feature, test_feature, SVEB_feature, VEB_feature
                else:
                    return train_feature, test_feature

if __name__ == '__main__':
    pass