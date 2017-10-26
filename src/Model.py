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
        self.max_iters = 1000
        self.lambd = 0.0 #1E-4
        self.batch_size = 100  # make this divisible by 2 please
        self.learn_rate = 2E-6
        self.epsilon = 1E-8
        self.im_len = self.loader.targ_im_h
        self.train_data = self.loader.get_train_data()
        self.test_data = self.loader.get_test_data()
        self.build_graph()
        self.sess = tf.Session()


    def build_conv_layer(self, input, in_depth, out_depth, width, height, max_pool=True):
    	"""
			pretty non-generalizable utility function.
    	"""
    	W = init_rand_weight([width, height, in_depth, out_depth]);
    	b = init_rand_bias(out_depth);
        conv = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='VALID') + b, name='features_1')
        if max_pool:
        	out = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
        	return out
        return conv

    def build_graph(self):

        # define graph
        if self.use_feed_dict:
            self.x = tf.placeholder(tf.float32, [None, self.im_len, self.im_len, self.loader.im_channels])
            self.y = tf.placeholder(tf.int32)
        else:
            self.x, self.y = self.loader.get_batch(self.batch_size)

        # center data
        mean, var = tf.nn.moments(self.x, axes=[0])
        self.x = (self.x - mean) / (tf.pow(var, 0.5) + self.epsilon)

        #depths = [1, 32, 64, 64]
        depths = [1, 64, 128, 128, 256]

        out1 = self.build_conv_layer(self.x, depths[0], depths[1], 10, 10);
        out2 = self.build_conv_layer(out1, depths[1], depths[2], 7, 7);
        out3 = self.build_conv_layer(out2, depths[2], depths[3], 4, 4);
        out4 = self.build_conv_layer(out3, depths[3], depths[4], 4, 4, False);

        flattened_len = 6 * 6 * depths[-1]
        flattened = tf.reshape(out4, [-1, flattened_len])

        fc1_len = 4096
        W_fc1 = init_rand_weight([flattened_len, fc1_len])
        b_fc1 = init_rand_bias([fc1_len])
        self.a_fc1 = tf.matmul(flattened, W_fc1) + b_fc1

        # first conv layer
        """W1 = init_rand_weight([3, 3, depths[0], depths[1]])
        b1 = init_rand_bias(depths[1])
        #out1 = tf.nn.relu(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1, name='features_1')
        conv1 = tf.nn.relu(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME') + b1, name='features_1')
        out1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')

        # second conv layer
        W2 = init_rand_weight([3, 3, depths[1], depths[2]])
        b2 = init_rand_bias(depths[2])
        #out2 = tf.nn.relu(tf.nn.conv2d(out1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2, name='features_2')
        conv2 = tf.nn.relu(tf.nn.conv2d(out1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2, name='features_2')
        out2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')

        # third conv layer
        W3 = init_rand_weight([3, 3, depths[2], depths[3]])
        b3 = init_rand_bias(depths[3])
        #out3 = tf.nn.relu(tf.nn.conv2d(out2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3, name='features_3')
        conv3 = tf.nn.relu(tf.nn.conv2d(out2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3, name='features_3')
        out3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')

        # 1st fc layer
        side_len = int(self.im_len / 8.0)
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
        
        W_d = init_rand_weight([fc2_len, 1])
        self.a_d = tf.squeeze(tf.matmul(self.a_f1, W_d))
        """

        W_final = init_rand_weight([fc1_len, 1])
        b_final = init_rand_bias(1)
        self.a_d = tf.squeeze(tf.matmul(self.a_fc1, W_final) + b_final)
        self.preds = tf.nn.sigmoid(self.a_d)
        self.labels = tf.cast(self.y, tf.float32) #tf.expand_dims(tf.cast(self.y, tf.int32), axis=1)

        # define loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.a_d, labels=self.labels))
        #self.loss += self.lambd * (tf.reduce_sum(tf.square(W_fc2)) + tf.reduce_sum(tf.square(W_fc1)))
        self.loss += self.lambd * (tf.reduce_sum(tf.square(W_fc1)))
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
        pred = self.sess.run(self.preds, feed_dict={self.x: [im]})
        return pred

    def get_evidence_arr(self, ims):
        preds = []
        for i in range(len(ims)):
            preds.append(self.get_evidence(ims[i]))
        return preds
