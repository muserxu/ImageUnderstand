# import tensorflow as tf
# from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(optimizer=tf.train.AdamOptimizer(), 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=5)
# arr = np.array([[  0.,  56.,  20.,  44.],
#                    [ 68.,   0.,  56.,   8.],
#                    [ 32.,  56.,   0.,  44.],
#                    [ 68.,  20.,  56.,   0.]])

# for i in arr:
#     print (i)
#     print (np.where(arr == i))

im = cv2.imread('detect1.jpg')
crop = im[255:734, 78:890]
plt.imshow(crop)
plt.show()