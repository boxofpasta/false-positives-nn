import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np
import functools
import shutil
from skimage.io import imread, imsave
import BatchLoader
import dataAugmentation


def transpose_py_list(l):
    return zip(*l)

def init_rand_weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev= 0.2))

def init_rand_bias(shape):
    return tf.Variable(tf.ones(shape))


class Model:

    def __init__(self):
        self.loader = BatchLoader.BatchLoader()
        self.np_train_data = None
        self.np_test_data = None
        self.cur_batch_iter = 0
        self.use_feed_dict = True
        self.max_iters = 1500
        self.lambd = 0.0 #1E-5
        self.batch_size = 80  # make this divisible by 2 please
        self.learn_rate = 2E-6
        self.im_len = self.loader.targ_im_h
        self.train_data = self.loader.get_train_data()
        self.test_data = self.loader.get_test_data()
        self.build_graph()
        self.sess = tf.Session()

    def build_graph(self):

        # define graph
        if self.use_feed_dict:
            self.x = tf.placeholder(tf.float32, [None, self.im_len, self.im_len, self.loader.im_channels])
            self.y = tf.placeholder(tf.int32)
        else:
            self.x, self.y = self.loader.get_batch(self.batch_size)

        # normalize data
        self.x = self.x - tf.reduce_mean(self.x, axis=0)
        transposed = tf.transpose(self.x)
        sqred_diffs = tf.squared_difference(tf.expand_dims(transposed, axis=1), tf.expand_dims(self.x, axis=0))
        stdev = tf.pow(tf.reduce_mean(sqred_diffs, axis=0), 0.5)
        self.x /= stdev

        depths = [1, 64, 128, 256]

        # first conv layer
        W1 = init_rand_weight([8, 8, depths[0], depths[1]])
        b1 = init_rand_bias(depths[1])
        out1 = tf.nn.relu(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1, name='features_1')
        #conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W1, strides=[1, 2, 2, 1], padding='SAME') + b1, name='features_1')
        #out1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')

        # second conv layer
        W2 = init_rand_weight([4, 4, depths[1], depths[2]])
        b2 = init_rand_bias(depths[2])
        out2 = tf.nn.relu(tf.nn.conv2d(out1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2, name='features_2')
        #conv2 = tf.nn.relu(tf.nn.conv2d(out1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2, name='features_2')
        #out2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')

        # third conv layer
        W3 = init_rand_weight([2, 2, depths[2], depths[3]])
        b3 = init_rand_bias(depths[3])
        out3 = tf.nn.relu(tf.nn.conv2d(out2, W3, strides=[1, 2, 2, 1], padding='SAME') + b3, name='features_3')
        #conv3 = tf.nn.relu(tf.nn.conv2d(out2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3, name='features_3')
        #out3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool_3')

        # 1st fc layer
        side_len = int(self.im_len / 4.0)
        flattened_len = side_len * side_len * depths[3]
        fc1_len = 512
        W_fc1 = init_rand_weight([flattened_len, fc1_len])
        b_fc1 = init_rand_bias([fc1_len])

        # reshape feature output, calculate activations
        a_fc1 = tf.nn.relu(tf.matmul(tf.reshape(out3, [-1, flattened_len]), W_fc1) + b_fc1)

        # final fc layer
        fc2_len = 64
        W_fc2 = init_rand_weight([fc1_len, fc2_len])
        b_fc2 = init_rand_bias([fc2_len])
        self.a_fc2 = tf.matmul(a_fc1, W_fc2) + b_fc2
        # self.preds = tf.nn.sigmoid(self.a_fc2)
        expanded_labels = tf.expand_dims(tf.cast(self.y, tf.float32), axis=1)

        # siamese pairings, same = 0, different = 1
        # note that batch_len / 2.0 != self.batch_len / 2.0 sometimes (near end of epoch, when not enough remains)
        batch_len = tf.shape(self.a_fc2)[0]
        self.dist = tf.abs(self.a_fc2[0 : batch_len/2, :] - self.a_fc2[batch_len/2 : batch_len, :])
        W_d = init_rand_weight([fc2_len, 1])
        self.a_d = tf.squeeze(tf.matmul(self.dist, W_d))
        self.preds = tf.nn.sigmoid(self.a_d)

        # reshape labels
        paired_y = tf.transpose(tf.reshape(expanded_labels, [2, -1]))
        zeros = tf.zeros([tf.shape(paired_y)[0]])
        self.labels = tf.cast(tf.equal(tf.abs(paired_y[:, 0] - paired_y[:, 1]), zeros), tf.float32)

        # define loss
        #self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(self.a_fc2, expanded_labels))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.a_d, self.labels))
        self.loss += self.lambd * (tf.reduce_sum(tf.square(W_fc2)) + tf.reduce_sum(tf.square(W_fc1)))
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss)

    def get_batch(self):
        start_pos = self.cur_batch_iter * self.batch_size

        # will go past the end
        if start_pos + self.batch_size >= len(self.np_train_data[0]):
            randindx = np.arange(len(self.np_train_data[0]))
            np.random.shuffle(randindx)
            self.np_train_data[0] = self.np_train_data[0][randindx]
            self.np_train_data[1] = self.np_train_data[1][randindx]
            self.cur_batch_iter = 0
            start_pos = 0

        end_pos = min(start_pos + self.batch_size, len(self.np_train_data[0]))
        batch = [self.np_train_data[0][start_pos:end_pos], self.np_train_data[1][start_pos:end_pos]]
        self.cur_batch_iter += 1

        return batch

    def train(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        self.np_train_data = self.sess.run(self.train_data)
        self.np_test_data = self.sess.run(self.test_data)

        train_loss_hist, test_loss_hist = [], []

        for i in range(self.max_iters):

            if self.use_feed_dict:
                batch = self.get_batch()
                _, train_loss, train_labels, train_preds = self.sess.run([self.train_step, self.loss, self.labels, self.preds], feed_dict={self.x: batch[0], self.y: batch[1]})
                #test_labels, test_preds = self.sess.run([self.y, self.preds], feed_dict={self.x: self.np_test_data[0], self.y: self.np_test_data[1]})
                #test_loss = self.sess.run([self.loss], feed_dict={self.x: self.np_test_data[0], self.y: self.np_test_data[1]})
            else:
                _, train_loss = self.sess.run([self.train_step, self.loss])
                labels, preds, ims = self.sess.run([self.y, self.preds, self.x])

            if i % 10 == 0:
                train_loss_hist.append(train_loss)
                #test_loss_hist.append(test_loss)
                #print train_labels
                #print np.squeeze(train_preds)
                print "at iteration : " +  str(i)
                print "with loss : " + str(train_loss)
                print ""

        """im, labels, preds = self.sess.run([self.x, self.y, self.preds], feed_dict={self.x: self.np_test_data[0], self.y: self.np_test_data[1]})
        print "TEST PREDICTIONS "
        im = np.reshape(im[-1], (self.im_len, self.im_len))
        plt.imshow(im)
        plt.show()
        print labels
        print np.squeeze(preds)"""
        coord.request_stop()
        coord.join(threads)
        #self.sess.close()

        return train_loss_hist

    def get_evidence(self, im):
        """ prediction is based on running the network on input and entire support set,
            and then finding the class im is most likely part of. """
        preds, labels = [], []
        for i in range(len(self.np_train_data[0])):
            pair = np.array([im, self.np_train_data[0][i]])
            pred = self.sess.run(self.preds, feed_dict={self.x: pair})
            label = self.np_train_data[1][i]
            preds.append(pred)
            labels.append(label)

        diff = np.abs(np.array(preds) - np.array(labels))
        false_evidence = np.sum(diff) / np.float(len(self.np_train_data[0]))
        true_evidence = 1.0 - false_evidence
        #print "true evidence: " + str(true_evidence)
        return true_evidence

    def get_evidence_arr(self, ims):
        preds = []
        for i in range(len(ims)):
            preds.append(self.get_evidence(ims[i]))
        return preds
