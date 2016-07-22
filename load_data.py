import glob

import numpy as np
import cv2

def glob_loader(globs):
    files_by_label = [glob.glob(g) for g in globs]
    images_with_label = \
        [(cv2.imread(f), label_id)
            for label_id, files in enumerate(files_by_label)
            for f in files]
    images = [image for image, label in images_with_label]
    labels = [label for image, label in images_with_label]
    return images, labels

def load_data():
    images, labels = glob_loader(['images/background_*.jpg', 'images/shark_*.jpg'])
    images_small = [cv2.resize(image, (32, 32)) for image in images]
    X = np.stack(images_small).astype(np.float32)
    Y = np.array(labels)
    return (X, Y), (X, Y)
