from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def smooth_curve(average_mae_history, factor=0.9):
    result = []
    for i in average_mae_history:
        if result:
            result.append(result[-1] * factor + i * (1 - factor))
        else:
            result.append(i)
    return result


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = np.mean(train_data, axis=0)
train_data -= mean
std = np.std(train_data, axis=0)
train_data /= std
test_data -= mean
test_data /= std
k = 4

num_val_samples = len(train_data) // k
num_of_epochs = 80
all_scores = []
all_mae_history = []

model = build_model()
model.fit(train_data, train_targets, epochs=num_of_epochs, batch_size=16)
print(model.evaluate(test_data, test_targets))



# validation step with K-validation
# for i in range(k):
#     val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
#     partial_train_data = np.concatenate((train_data[:i * num_val_samples],  train_data[(i+1) * num_val_samples:]),
#                                         axis=0)
#     partial_train_targets = np.concatenate((train_targets[:i * num_val_samples],
#                                             train_targets[(i+1) * num_val_samples:]), axis=0)
#     model = build_model()
#     history = model.fit(partial_train_data, partial_train_targets, epochs=num_of_epochs, batch_size=1,
#                         validation_data=(val_data, val_targets))
#     all_mae_history.append(history.history['val_mean_absolute_error'])
#     # val_mse, val_mae = model.evaluate(val_data, val_targets)
#     # all_scores.append(val_mae)
# average_mae_history = [np.mean([x[i] for x in all_mae_history])for i in range(num_of_epochs)]
# plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
# plt.xlabel("Epochs")
# plt.ylabel("Validation MAE")
# plt.show()
#
# smooth_points = smooth_curve(average_mae_history[20:])
# plt.plot(range(1, len(smooth_points)+1), smooth_points)
# plt.xlabel("Epochs")
# plt.ylabel("Validation MAE")
# plt.show()

