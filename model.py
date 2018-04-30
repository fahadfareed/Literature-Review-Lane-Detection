import tensorflow as tf
import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tf.placeholder(tf.float32, shape=[None, 170, 300, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 403])

"""
i1: 17x30x3
k1: 5x5x3x24 sd1: 1

i2: 13x26x24
k2: 5x5x24x36 sd2: 1

i3: 9x22x36
k3: 5x5x36x48 sd3: 1

i4: 5x18x48
k4: 3x3x48x64 sd4: 2

i5: 3x16x64
k5: 3x3x64x64 sd4: 1

i6: 1x14x64 -> 896
"""

"""
i1: 40x71x3
k1: 5x5x3x24 sd1: 2

i2: 18x34x24
k2: 5x5x24x36 sd2: 1

i3: 14x30x36
k3: 5x5x36x48 sd3: 1

i4: 10x26x48
k4: 3x3x48x64 sd4: 2

i5: 4x12x64
k5: 3x3x64x64 sd4: 1

i6: 2x10x64 -> 1280
"""

"""
i1: 170x300x3
k1: 5x5x3x24 sd1: 2

i2: 83x148x24
k2: 5x5x24x36 sd2: 2

i3: 40x72x36
k3: 5x5x36x48 sd3: 2

i4: 18x34x48
k4: 3x3x48x64 sd4: 2

i5: 8x16x64
k5: 3x3x64x64 sd4: 2

i6: 3x7x64 -> 1344
"""

"""
i1: 80x142x3
k1: 5x5x3x24 sd1: 2

i2: 38x69x24
k2: 5x5x24x36 sd2: 2

i3: 17x33x36
k3: 5x5x36x48 sd3: 2

i4: 7x15x48
k4: 3x3x48x64 sd4: 1

i5: 5x13x64
k5: 3x3x64x64 sd4: 1

i6: 3x11x64 -> 2112
"""

x_image = x

#first convolutional layer
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

#second convolutional layer
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#third convolutional layer
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)

#fourth convolutional layer
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 2) + b_conv4)

#fifth convolutional layer
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 2) + b_conv5)

#FCL 1
W_fc1 = weight_variable([1344, 403])
b_fc1 = bias_variable([403])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1344])

y = tf.matmul(h_conv5_flat, W_fc1) + b_fc1

keep_prob = tf.placeholder(tf.float32)

#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
