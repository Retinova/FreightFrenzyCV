import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from cnn.FreightFrenzyCNN import UltimateGoalCNN

batch_size = 16
epochs = 7

rel_img_path = 'C:/Users/Robotics3/PycharmProjects/pythonProject/dataset/raw_split'
num_train_images = 295
num_test_images = 30

# train_data = np.empty((num_train_images, 3264, 2448, 3), dtype='uint8')
train_data = np.empty((num_train_images, 2448, 2448, 3), dtype='uint8')
train_labels = np.array([], dtype='uint8')

# test_data = np.empty((num_test_images, 3264, 2448, 3), dtype='uint8')
test_data = np.empty((num_test_images, 2448, 2448, 3), dtype='uint8')
test_labels = np.array([], dtype='uint8')

# load images
labels = ['a', 'b', 'c']  # using this list guarantees that the images will be loaded in a consistent order since os.listdir() lists an arbitrary order
lbl_count = 0
img_count = 0
print('Loading training images...')
path = os.path.join(rel_img_path, 'training')
for label in labels:
    temp = os.path.join(path, label)
    for file in os.listdir(temp):
        img = image.load_img(os.path.join(temp, file))
        img = np.asarray(img)
        # remove later
        img = img[408:2856]
        train_data[img_count] = img
        train_labels = np.append(train_labels, lbl_count)
        img_count += 1
        print('\r' + str(img_count) + '/' + str(num_train_images) + ' loaded...', end='')
    lbl_count += 1

print('\nTraining images loaded!\nLoading testing images...')
lbl_count = 0
img_count = 0
path = os.path.join(rel_img_path, 'testing')
for label in labels:
    temp = os.path.join(path, label)
    for file in os.listdir(temp):
        img = image.load_img(os.path.join(temp, file))
        img = np.asarray(img)
        # remove later
        img = img[408:2856]
        test_data[img_count] = img
        test_labels = np.append(test_labels, lbl_count)
        img_count += 1
        print('\r' + str(img_count) + '/' + str(num_test_images) + ' loaded...', end='')
    lbl_count += 1
print('\nTesting images loaded!')

# reformat data and labels
train_labels = np.reshape(train_labels, len(train_labels))
test_labels = np.reshape(test_labels, len(test_labels))

(x_train, y_train) = (train_data, train_labels)
(x_test, y_test) = (test_data, test_labels)

# reformat labels into one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

print("Loaded all data.")

# training
model = UltimateGoalCNN(x_train.shape[1:]).model
loss_fn = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.Adam()

model.summary()

# stop/continue idk
chk = input('Continue? (y/n)\n')
if chk.lower() != 'y':
    raise ValueError

model.compile(optimizer, loss_fn, metrics=['accuracy'])  # compile the model with its optimizer and loss functions (metrics are displayed during training)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)  # train the model

model.save('C:/Users/Robotics3/PycharmProjects/pythonProject/saved_models/v1.1', include_optimizer=True, overwrite=True)  # save the entire model (arch., weights, etc.)

# evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ' + str(scores[0]))
print('Test acc.: ' + str(scores[1]))
