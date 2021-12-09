from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
# from tensorflow.keras import utils
# import os  # for fixing plot_model


class EpicCNN:
    def __init__(self, input_shape):
        self.model = Sequential()

        # preprocessing layers
        self.model.add(layers.Input(shape=input_shape))
        # self.model.add(preprocessing.CenterCrop(self.model.input_shape[2], self.model.input_shape[2], name='crop_input'))  # make the image square
        self.model.add(preprocessing.Resizing(360, 640, interpolation='bilinear', name='downscale'))
        self.model.add(preprocessing.Rescaling(scale=1.0/255, name='pixel_val_rescale'))  # scale pixel values from [0, 255] to [0, 1]

        # data augmentation layers (only active during training)
        # self.model.add(preprocessing.RandomFlip(mode='horizontal', name='horiz_flip_aug'))  # randomly flips training images horizontally
        # self.model.add(preprocessing.RandomRotation(factor=(-1.0/24, 1.0/24), fill_mode='nearest', name='rotation_aug'))  # randomly rotates training images within [-pi/12, pi/12]
        # self.model.add(preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.1, fill_mode='nearest', name='translation_aug'))  # randomly translates images

        # actual learning layers
        self.model.add(layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))

        self.model.add(layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', data_format='channels_last', activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))
        self.model.add(layers.MaxPool2D(pool_size=2))

        self.model.add(layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same', data_format='channels_last', activation='relu'))
        self.model.add(layers.Dropout(rate=0.1))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=576, activation='relu'))
        self.model.add(layers.Dropout(rate=0.4))
        self.model.add(layers.Dense(units=3, activation='softmax'))


# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # for fixing plot model
# test = EpicCNN((480, 270, 3))
# test.model.summary()
# utils.plot_model(test.model, 'v1_no_preproc.png', True, dpi=192)
