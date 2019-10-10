from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


def vectorize_sequence(data, dimension=10000):
    results = np.zeros((len(data), dimension))
    for ind, cols in enumerate(data):
        results[ind, cols] = 1
    return results


(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = vectorize_sequence(x_train)
x_test = vectorize_sequence(x_test)
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
x_val = x_train[:1000]
partial_x = x_train[1000:]
y_val = y_train[:1000]
partial_y = y_train[1000:]
history = model.fit(partial_x, partial_y, epochs=10, batch_size=512, validation_data=(x_val, y_val))

# loss and accuracy plot
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()
plt.clf()
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.show()
print(model.evaluate(x_test, y_test))
predictions = model.predict(x_test)
print(np.argmax(predictions[45]))
print(y_test[45])

