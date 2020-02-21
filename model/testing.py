import tensorflow as tf
from utils import rgb_to_lab, lab_to_rgb
from model import ColorizeGrayScale
from config import reg
from data import normalize_rgb
import skimage.io as io
import skimage.transform as transform
import skimage.color as color
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

# Do some code, e.g. train and save model

K.clear_session()
model = ColorizeGrayScale(l2_reg=reg)

model.load_weights('../model_weights/model.53')

# prepare the input
rgb = io.imread('../images/2007_000027.jpg')

# resize the image
input_1 = transform.resize(rgb, (224, 224)).astype(np.float32)
input_2 = transform.resize(rgb, (229, 229)).astype(np.float32)
# conver input_1 to lab
lab_img = rgb_to_lab(input_1)

l_channel = lab_img[:, :, 0]

# convert to grayscale
input_1 = color.rgb2gray(input_1)

input_2 = color.rgb2gray(input_2)
input_2 = np.stack([input_2] * 3, axis=-1)

# Normalize the image [-1, 1]
input_1 = normalize_rgb(input_1)
input_2 = normalize_rgb(input_2)

# add batch dimension
input_1 = np.expand_dims(np.expand_dims(input_1, axis=0), axis=-1)
input_2 = np.expand_dims(input_2, 0)

inputs = {'input_1': input_1, 'input_2': input_2}

# run the model
output = model(inputs)

output = np.squeeze(output.numpy(), axis=0)
print(np.unique(output))

# Multiply output by 128
output = output * 128

# Add grayscale channel with output
lab_result = tf.stack([l_channel, output[:, :, 0], output[:, :, 1]], axis=-1)

# convert lab result to rgb
rgb_colorized = lab_to_rgb(lab_result).numpy()

# plot the result
plt.imshow(rgb_colorized[:, :, 0])
plt.show()
