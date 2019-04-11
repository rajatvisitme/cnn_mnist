# importing required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

#Loading datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#adding layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=x_train[0].shape))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam',
			loss = 'sparse_categorical_crossentropy',
			metrics = ['accuracy'])

#Training the model
model.fit(x_train, y_train, epochs = 5)

#Returns the loss value & accuracy for the model in test mode
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

# saving the model
model.save('cnn_mnist.model')

# loading the model
new_model = tf.keras.models.load_model('cnn_mnist.model')
predictions = new_model.predict([x_test])

#predictions for the first image of test data
print(predictions[0]) 

# highest predicted value for the first image (0-7)
print(np.argmax(predictions[0]))	

# plotting the actual image
plt.imshow(x_test[0])
plt.show()
