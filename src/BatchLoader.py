import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np
import functools
import shutil
from skimage.io import imread, imsave


def one_use(func):
    """ tf graph construction functions that should only
        be called once will find this decorator useful """
    attribute = "_cache_" + func.__name__

    @property
    @functools.wraps(func)
    def decorated(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return decorated


class BatchLoader:

    def __init__(self):
        
        # images will be warped to these dimensions for inference / training
        self.im_channels = 1
        self.targ_im_h = 105#40
        self.targ_im_w = 105#40
        self.train_path = "../images/augmented_train/"
        self.test_path = "../images/test/"
        self.im = None
        self.train_data_size = 0
        self.test_data_size = 0
        self.init_queues


    @one_use
    def init_queues(self):
        print "initiating file name queues ..."

        # note that these are all tensors
        self.train_label, self.train_im, self.train_data_size = self.get_queues_from_directory(self.train_path)
        self.test_label, self.test_im, self.test_data_size = self.get_queues_from_directory(self.test_path)


    def get_queues_from_directory(self, dir_path):
        names = os.listdir(dir_path)
        names = sorted([name for name in names if name.endswith(".jpg")])
        labels = [int(name[:4] == "true") for name in names]

        im_names = [dir_path + name for name in names]
        data_size = len(im_names)

        im_names = tf.convert_to_tensor(im_names, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        queue = tf.train.slice_input_producer([im_names, labels], shuffle=False)

        file_content = tf.read_file(queue[0])
        im = tf.image.decode_jpeg(file_content, channels=self.im_channels)

        return queue[1], tf.image.resize_images(im, [self.targ_im_h, self.targ_im_w]) / 255.0, data_size


    def get_batch(self, batch_size=30):
        return tf.train.batch([self.train_im, self.train_label], batch_size)

    def get_train_data(self):
        return tf.train.batch([self.train_im, self.train_label], self.train_data_size)

    def get_test_data(self):
        return tf.train.batch([self.test_im, self.test_label], self.test_data_size)