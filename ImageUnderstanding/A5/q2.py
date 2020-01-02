import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print (train_images.shape)
# print (train_labels)
# plt.figure()
# plt.imshow(train_images[0])
# plt.show()
# plt.colorbar()
# plt.grid(False)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

# q2c
unique, counts = np.unique(train_labels, return_counts=True)
d = dict(zip(unique, counts))
unique, counts = np.unique(test_labels, return_counts=True)
e = dict(zip(unique, counts))
for key,value in e.items():
    if key in d:
        d[key] += value
plt.bar(list(d.keys()),list(d.values()), color='b')
t1 = np.arange(-1, 10, 1)
plt.xticks(range(len(class_names)),class_names)
plt.show()


# q2d,g
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, batch_size = 64)

loss = list(history.history['loss'])
plt.figure(1)
plt.subplot(311)
plt.plot(range(len(loss)), loss, 'ro-')
plt.axis([-0.5, 19.5, 0.0, 1.0])
plt.xlabel('Epoch')
plt.ylabel('loss')

acc = list(history.history['acc'])
plt.subplot(312)
plt.plot(range(len(acc)), acc, 'ro-')
plt.axis([-0.5, 19.5, 0.0, 1.0])
plt.xlabel('Epoch')
plt.ylabel('accuary')

plt.show()


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc, 'testloss:', test_loss)

# q2e
predictions = model.predict(test_images)

# def plot_image(i, predictions_array, true_label, img):
#   predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
  
#   plt.imshow(img, cmap=plt.cm.binary)

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'
  
#   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array[i], true_label[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0, 1]) 
#   predicted_label = np.argmax(predictions_array)
 
#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')

# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
# plt.show()


#q2g
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch = [8,16,32,64]
train_loss=[]
train_acc=[]
valid_loss=[]
valid_acc=[]
for i in batch:
    history = model.fit(train_images, train_labels, epochs=5, batch_size = i)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    train_loss.append(list(history.history['loss'])[-1])
    train_acc.append(list(history.history['acc'])[-1])
    valid_loss.append(test_loss)
    valid_acc.append(test_acc)


plt.figure(1)
plt.subplot(211)
plt.plot(batch, train_loss, 'ro-')
plt.axis([7.5, 64.5, 0.15, 0.4])
plt.xlabel('baych')
plt.ylabel('loss')

plt.subplot(212)
plt.plot(batch, train_acc, 'ro-')
plt.axis([7.5, 64.5, 0.8, 1.0])
plt.xlabel('batch')
plt.ylabel('accuary')


plt.subplot(413)
plt.plot(batch, valid_loss, 'ro-')
plt.axis([7.5, 64.5, 0.2, 0.4])
plt.xlabel('batch')
plt.ylabel('valid_loss')

plt.subplot(414)
plt.plot(batch, valid_acc, 'ro-')
plt.axis([7.5, 64.5, 0.8, 1.0])
plt.xlabel('batch')
plt.ylabel('valid_accuary')


plt.show()


