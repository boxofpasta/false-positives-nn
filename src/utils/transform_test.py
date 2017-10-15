import os
import sys
from skimage import io
from skimage import transform as tf

""" 
mostly just for testing transformation effects.
https://stackoverflow.com/questions/24191545/skewing-or-shearing-an-image-in-python
"""

dir = sys.argv[1]

# Load the image as a matrix
image = io.imread(dir)
im_width = len(image[1])

# Create Afine transform
afine_tf = tf.AffineTransform(shear=0.4, translation=[0.2 * im_width, 0])

# Apply transform to image data
modified = tf.warp(image, inverse_map=afine_tf)

# Display the result
io.imshow(modified)
io.show()