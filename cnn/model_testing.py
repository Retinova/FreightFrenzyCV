import keras.models

from FreightFrenzyCNN import EpicCNN
import numpy as np
import tensorflow.keras
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt
import random

# dataset arrays
test_data = np.empty((600, 360, 640, 3), dtype='uint8')
test_labels = np.array([], dtype='uint8')

rel_img_path = 'C:/Users/Robotics3/PycharmProjects/FreightFrenzyCV/dataset/split'

# load images
labels = ['center', 'left', 'right']  # using this list guarantees that the images will be loaded in a consistent order since os.listdir() lists an arbitrary order
lbl_count = 0
img_count = 0
path = os.path.join(rel_img_path, 'testing')
print('Loading testing images...')
for label in labels:
    temp = os.path.join(path, label)
    for file in os.listdir(temp):
        img = image.load_img(os.path.join(temp, file))
        img = np.asarray(img)
        test_data[img_count] = img
        test_labels = np.append(test_labels, lbl_count)
        img_count += 1
        print('\r' + str(img_count) + '/' + '600' + ' loaded...', end='')
    lbl_count += 1
print('\nTesting images loaded!')

test_labels = np.reshape(test_labels, len(test_labels))
test_labels = utils.to_categorical(test_labels, num_classes=3)

print("Loaded all data.")

model = keras.models.load_model('saved_models/v1.0/')

print("Model loaded.")

model.summary()

num_tests = int(input('\nHow many images would you like to test?\n'))
picks = random.sample(range(0, len(test_labels)), num_tests)

show_ims = True if input('Display images? (y/n)\n').lower() == 'y' else False

for i in picks:
    if show_ims:
        plt.imshow(test_data[i])
        # mngr = plt.get_current_fig_manager()
        # geom = mngr.window.geometry()
        # x, y, dx, dy = geom.getRect()
        # mngr.window.setGeometry(700, 400, dx, dy)
        plt.show()
    guess = model.predict(np.expand_dims(test_data[i], axis=0))
    print('Guesses:')
    print('Center: ', guess[0][0])
    print('Left: ', guess[0][1])
    print('Right: ', guess[0][2])
    print('Correct label: ', labels[np.argmax(test_labels[i])])
