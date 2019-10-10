from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers


def vectorize_sequence(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for i, s in enumerate(sequences):
        result[i, s] = 1
    return result

((train_data, train_labels), (test_data, test_labels)) = imdb.load_data(num_words=1000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
first_review = ' '.join([ reverse_word_index.get(i-3, '?') for i in train_data[0]])
print(first_review)
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_tes= np.asarray(test_labels).astype('float32')
x_val = x_train[10000:]
partial_x_train = x_train[:10000]
y_val = y_train[10000:]
partial_y_train = y_train[:10000]
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
k = history.history
loss_values = k['loss']
val_loss_values = k['val_loss']


# model regularizer

model2 = models.Sequential()
model2.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.001), input_shape=(10000, )))
model2.add(layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history2 = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
k2 = history2.history
large_model_val_loss = k2['val_loss']
# comparison of loss between small and large model
length_epochs = range(1, len(loss_values)+1)
plt.plot(length_epochs, val_loss_values, 'bo', label="Normal Model")
plt.plot(length_epochs, large_model_val_loss, 'b+', label="Regularizer Model")
plt.xlabel("Epochs")
plt.ylabel("Model")
plt.show()







# # large model with more layers
#
# model2 = models.Sequential()
# model2.add(layers.Dense(512, activation='relu', input_shape=(10000, )))
# model2.add(layers.Dense(512, activation='relu'))
# model2.add(layers.Dense(1, activation='sigmoid'))
# model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# history2 = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
# k2 = history2.history
# large_model_val_loss = k2['val_loss']
# # comparison of loss between small and large model
# length_epochs = range(1, len(loss_values)+1)
# plt.plot(length_epochs, val_loss_values, 'bo', label="Normal Model")
# plt.plot(length_epochs, large_model_val_loss, 'b+', label="Large Model")
# plt.xlabel("Epochs")
# plt.ylabel("Model")
# plt.show()

# small model with less layers
#
# model2 = models.Sequential()
# model2.add(layers.Dense(4, activation='relu', input_shape=(10000, )))
# model2.add(layers.Dense(4, activation='relu'))
# model2.add(layers.Dense(1, activation='sigmoid'))
# model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# history2 = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
# k2 = history2.history
# small_model_val_loss = k2['val_loss']
# # comparison of loss between small and large model
# length_epochs = range(1, len(loss_values)+1)
# plt.plot(length_epochs, val_loss_values, 'bo', label="Normal Model")
# plt.plot(length_epochs, small_model_val_loss, 'b+', label="Small Model")
# plt.xlabel("Epochs")
# plt.ylabel("Model")
# plt.show()
#
# epochs = range(1, len(loss_values)+1)
# plt.plot(epochs, loss_values, 'bo', label="Training Loss")
# plt.plot(epochs, val_loss_values, 'b', label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()
# acc_values = k['acc']
# val_acc_values = k['val_acc']
# epochs = range(1, len(acc_values)+1)
# plt.plot(epochs, acc_values, 'bo', label="Training Accuracy")
# plt.plot(epochs, val_acc_values, 'b', label="Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.show()
# pass
