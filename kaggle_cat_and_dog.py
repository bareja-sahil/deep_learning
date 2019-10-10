import os
import shutil
# downloaded training images
from keras import models, layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

downloaded_train_path = '/Users/sahil-k/Downloads/dogs-vs-cats/train'
current_directory = os.path.dirname(__file__)
# train directory
train_path = os.path.join(current_directory, 'train')
cats_train_path = os.path.join(train_path, 'cats')
dogs_train_path = os.path.join(train_path, 'dogs')
# test directory
test_path = os.path.join(current_directory, 'test')
cats_test_path = os.path.join(test_path, 'cats')
dogs_test_path = os.path.join(test_path, 'dogs')
# validate directory
validate_path = os.path.join(current_directory, 'validate')
cats_validate_path = os.path.join(validate_path, 'cats')
dogs_validate_path = os.path.join(validate_path, 'dogs')

# directory creation
# os.makedirs(train_path)
# os.makedirs(cats_train_path)
# os.makedirs(dogs_train_path)
# os.makedirs(test_path)
# os.makedirs(cats_test_path)
# os.makedirs(dogs_test_path)
# os.makedirs(validate_path)
# os.makedirs(cats_validate_path)
# os.makedirs(dogs_validate_path)


# # copy cats to train, test and validate
# for i in range(1000):
#     cat_fl = 'cat.%s.jpg' % i
#     dog_fl = 'dog.%s.jpg' % i
#     src1 = os.path.join(downloaded_train_path, cat_fl)
#     dst1 = os.path.join(cats_train_path, cat_fl)
#     shutil.copyfile(src1, dst1)
#     src2 = os.path.join(downloaded_train_path, dog_fl)
#     dst2 = os.path.join(dogs_train_path, dog_fl)
#     shutil.copyfile(src2, dst2)
# for i in range(1000, 1500):
#     cat_fl = 'cat.%s.jpg' % i
#     dog_fl = 'dog.%s.jpg' % i
#     src1 = os.path.join(downloaded_train_path, cat_fl)
#     dst1 = os.path.join(cats_test_path, cat_fl)
#     shutil.copyfile(src1, dst1)
#     src2 = os.path.join(downloaded_train_path, dog_fl)
#     dst2 = os.path.join(dogs_test_path, dog_fl)
#     shutil.copyfile(src2, dst2)
# for i in range(1500, 2000):
#     cat_fl = 'cat.%s.jpg' % i
#     dog_fl = 'dog.%s.jpg' % i
#     src1 = os.path.join(downloaded_train_path, cat_fl)
#     dst1 = os.path.join(cats_validate_path, cat_fl)
#     shutil.copyfile(src1, dst1)
#     src2 = os.path.join(downloaded_train_path, dog_fl)
#     dst2 = os.path.join(dogs_validate_path, dog_fl)
#     shutil.copyfile(src2, dst2)

# build up your own CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# print(model.summary())

train_datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1/255.)
train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validate_path,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=validation_generator, validation_steps=50)
model.save('cats_and_dogs_small_2.h5')
