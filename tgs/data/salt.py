import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os
from tqdm import tqdm


HEIGHT = 128
WIDTH = 128


class Image:
    def __init__(self, img_id, path):
        self.path = path
        self.img_id = img_id

        self.img = imread(self.path + "/images/" + img_id, as_gray=True)

        if os.path.exists(self.path + "/masks"):
            self.mask = imread(self.path + "/masks/" + img_id, as_gray=True)
        else:
            self.mask = None

    def get_img(self, size=(HEIGHT, WIDTH)):
        if self.mask is None:
            return resize(self.img, output_shape=size), None
        else:
            return resize(self.img, size), resize(self.mask, output_shape=size)


class ImageSet:
    def __init__(self, images, im_height, im_width, im_chan):
        self.X = np.zeros((len(images), im_height, im_width, im_chan), dtype=np.float32)
        self.y = np.zeros((len(images), im_height, im_width, 1), dtype=np.float32)

        self.n = len(images)

        k = 0
        for image in tqdm(images):
            self.X[k, ..., 0], self.y[k, ..., 0] = image.get_img()  # ... to let place for features if needed
            k =+ 1

    def get_x_y(self):
        return self.X, self.y

