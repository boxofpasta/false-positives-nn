import tensorflow as tf
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np
import functools
import shutil
from skimage.io import imread, imsave
import Model
import dataAugmentation

if __name__ == '__main__':
    #dataAugmentation.augment_data("../images/original_train/", "../images/augmented_train/")

    m = Model.Model()
    train_loss_hist = m.train()
    evidences = m.get_evidence_arr(m.np_test_data[0])
    train_evidences = m.get_evidence_arr(m.np_train_data[0])

    print 'TEST'
    print evidences
    print m.np_test_data[1]
    print ' '
    print 'TRAIN'
    print train_evidences
    print m.np_train_data[1]

    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist, label="train loss")
    #plt.plot(np.arange(len(test_loss_hist)), test_loss_hist, label="test loss")
    plt.legend(loc='upper right', fontsize=10)
    plt.show()