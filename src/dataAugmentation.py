import tensorflow as tf
from skimage import transform
from scipy import misc
from matplotlib import pyplot as plt
import os
import numpy as np


def augment_data(in_folder_path, out_folder_path):
    """ adds rotations, shears, flips to existing images, and saves
        the new ones to the same folder """

    num_rots = 10
    cwd = os.getcwd()
    names = os.listdir(in_folder_path)
    names = [name for name in names if name.endswith(".jpg")]
    transformed = []

    for i in range(len(names)):
        original = plt.imread(in_folder_path + names[i])
        flipped = np.fliplr(original)

        for rot in range(num_rots - 1):
            deg = rot * 360.0 / num_rots + 360.0 / num_rots
            original_rotated = transform.rotate(original, deg)
            flipped_rotated = transform.rotate(flipped, deg)
            transformed.append([names[i][:-4] + "_" + str(rot) + ".jpg", original_rotated])
            transformed.append([names[i][:-4] + "_" + str(rot) + "f" + ".jpg", flipped_rotated])

    for i in range(len(transformed)):
        plt.imsave(out_folder_path + transformed[i][0], transformed[i][1])