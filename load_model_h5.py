from keras.models import load_model
import os
history = load_model(os.path.join(os.path.dirname(__file__), 'cats_and_dogs_small_1.h5'))
pass