import tensorflow as tf
import scipy.misc
import model
import cv2
from subprocess import call
import numpy as np
import ast
import csv
from os import listdir as ls
import time as t
import matplotlib.pyplot as plt
import utils
import heapq

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "Models/170x300/save/model.ckpt")

#times = np.arange(40, 86, 1/60.0)
#image_paths = [x for x in ls('images_new') if x[-5:] == '.jpeg']

#for path in image_paths:

def remove_outliers(coords):
    elements = np.array([c[0] for c in coords])
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    return  [c for c in coords if (c[0] > mean - 2 * sd and c[0] < mean + 2* sd)]

def apply_polynomial(coords, order=2):
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    z = np.polyfit(y, x, order)
    p = np.poly1d(z)

    return [(p(y), y) for y in utils.y_]

path='40.0.jpeg'

img = scipy.misc.imread('images/' + path)
img_original = img
img = scipy.misc.imresize(img, [170, 300]) / 255.0 - 0.5

data = np.array(utils.read_csv('data_grid.csv'))

t1 = t.time()
out = model.y.eval(feed_dict={model.x: [img], model.keep_prob: 1.0})
out = out[0] + np.max(out[0])

for row in data:
	if(row[0] == path[:-5]):
		actual = np.array(ast.literal_eval(data[0][1]))

print(heapq.nlargest(23, out)[22])
thresh = heapq.nlargest(31, out)[30]

out = [o if o > thresh else 0 for o in out]
pred = []

for i in range(len(out) - 1):
	if (out[i] > 0 and out[i+1]>0) or (out[i] > 0 and out[i+2]>0):
		pass
	else:
		pred.append(out[i])

thresh = heapq.nlargest(23, pred)[22]
pred = np.array([1 if p > thresh else 0 for p in pred])

print('Actual: ', list(np.where(actual==1)[0]))
print('Predicted: ', list(np.where(pred==1)[0]))
print('Time taken: ', t.time() - t1)

sizes_x = [16]

for i in range(1, len(utils.y_)):
	ratio = (12 * (1 - utils.y_[i]/utils.y_[i-1])) * int(1280/16.0)
	sizes_x.append(int(1280/ratio))

sizes_x = np.cumsum(sizes_x)
print('Sizes: ', sizes_x)

coords = []
for i, o in enumerate(actual):
	if o == 1:
		arr = np.array(sizes_x - i)
		arr = np.array([1000 if a<0 else a for a in arr])
		idx = arr.argmin()

		if idx == 0:
			spacing = 1280.0 / 16
		else:
			spacing = 1280.0 / (sizes_x[idx] - sizes_x[idx-1])
		
		print('Spacing: ', spacing)
		x = sizes_x[idx] - i
		print('Distance: ', x)
		coords.append((1280 - (x+0.5)*spacing, utils.y_[idx]))

coords_left = [c for i, c in enumerate(coords) if i%2==0]
coords_right = [c for i, c in enumerate(coords) if i%2!=0]

#print(coords_left)

coords_left = apply_polynomial(remove_outliers(coords_left))
coords_right = apply_polynomial(remove_outliers(coords_right))

coords_left_x = [c[0] for c in coords_left]
coords_right_x = [c[0] for c in coords_right]
coords_left_y = [c[1] for c in coords_left]
coords_right_y = [c[1] for c in coords_right]

plt.imshow(img_original)
plt.scatter(x=coords_left_x, y=coords_left_y, c='r', s=100)
plt.scatter(x=coords_right_x, y=coords_right_y, c='g', s=100)
plt.show()
"""
out_temp = []

for o in out:
	out_temp.append(o)

out_temp.sort()
out_temp.reverse()

for i, o in enumerate(out):
	if(o in out_temp[0:19]):
		out[i] = 1
	else:
		out[i] = 0

data = utils.read_csv('data_grid.csv')

actual = []
for row in data:
	if(row[0] == path[:-5]):
		actual = ast.literal_eval(data[0][1])

i_actual = []
for i, a in enumerate(actual):
	if(a == 1):
		i_actual.append(i)

img = img_original

two_d_out = [out[i:i+32] for i in range(0, len(out), 32)]

left_x_scaled = []
right_x_scaled = []

for o in two_d_out:
	indexes = []
	for i, x in enumerate(o):
		if(x == 1):
			indexes.append(i)

	if( not len(indexes) == 0):
		left_x_scaled.append(indexes[0])
		right_x_scaled.append(indexes[len(indexes) - 1])
	else:
		if not len(left_x_scaled) == 0:
			left_x_scaled.append(left_x_scaled[len(left_x_scaled) - 1])
		if not len(right_x_scaled) == 0:
			right_x_scaled.append(right_x_scaled[len(right_x_scaled) - 1])

pts_left_x = []
pts_right_x = []

for x_l, x_r in zip(left_x_scaled, right_x_scaled):
	pts_left_x.append(x_l*40 + 20)
	pts_right_x.append(x_r*40 + 20)

img_size = np.shape(img)

pts_left_x = utils.remove_outliers(pts_left_x)
pts_right_x = utils.remove_outliers(pts_right_x)

pts_left_y = [720*(1 - i/32.0 - 1/64.0) for i in range(0, len(pts_left_x))]
pts_right_y = [720*(1 - i/32.0 - 1/64.0) for i in range(0, len(pts_right_x))]

pts_left_x = utils.apply_polynomial(pts_left_x, pts_left_y)
pts_right_x = utils.apply_polynomial(pts_right_x, pts_right_y)

pts_left = [(x, y) for x, y in zip(pts_left_x, utils.y_)]
pts_right = [(x, y) for x, y in zip(pts_right_x, utils.y_)]

coordinates = np.concatenate([pts_left, pts_right])
grid = utils.create_grid(coordinates)

true_vals = []
preds = []

i_grid = []

for i, a in enumerate(grid):
	if(a == 1):
		i_grid.append(i)

for i in range(len(i_actual)-1):
	if (i_actual[i+1]-i_actual[i] != 1):
		true_vals.append(i_actual[i])

for i in range(len(i_grid)-1):
	if (i_grid[i+1]-i_grid[i] != 1):
		preds.append(i_grid[i])

a = [(x-y)**2 for x, y in zip(i_actual, i_grid)]
mse = np.sqrt(np.sum(a))/float(len(a))
print('Mean Squared Error: ', mse)
print("Time taken: ", t.time() - t1)

print(true_vals)
print(preds)

plt.imshow(img)
plt.scatter(x=pts_left_x, y=utils.y_, c='r', s=100)
plt.scatter(x=pts_right_x, y=utils.y_, c='b', s=100)

plt.text(x=800, y=200, s='Mse: '+ str(mse))
plt.text(x=800, y=300, s=path)


plt.show()

"""