from __future__ import division, print_function, absolute_import

# Pre-requisite ...
#   pip install tflearn

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

num_classes = 2
image_size = (64, 64)

import load_data
(X, Y), (X_test, Y_test) = load_data.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, num_classes)
Y_test = to_categorical(Y_test, num_classes)


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=180.)
img_aug.add_random_crop(image_size, padding=6)

def build_network(image_size, batch_size=None, n_channels=3):
    network = input_data(shape=[batch_size, image_size[0], image_size[1], n_channels],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
    network = conv_2d(network, 16, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.0001)

    return network

def build_model():
    network = build_network(image_size=image_size)
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model

if __name__ == '__main__':
    model = build_model()

    model.fit(X, Y, n_epoch=2, shuffle=True, validation_set=(X_test, Y_test),
              show_metric=True, batch_size=1, run_id='detect_cnn')

    model.save('detect_cnn.tflearn')
