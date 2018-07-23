from tgs.models.UNet import get_unet
from keras.callbacks import ModelCheckpoint
from tgs.data.images import HEIGHT, WIDTH, Image, ImageSet
from tgs.data.split import get_train_val_ids

from sklearn.model_selection import train_test_split

import os
import random as rdm
# TODO MLFlow


if __name__ == '__main__':
    model = get_unet((HEIGHT, WIDTH, 1))
    model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

    imgs_train = list()
    imgs_val = list()

    img_path = "data/raw/train"

    # TODO Later - Cache ImageSet object
    train_ids, val_ids = get_train_val_ids(img_path + "/images/")
    for img_id in os.listdir(img_path + "/images/"):
        if img_id in train_ids:
            imgs_train.append(Image(img_id, img_path))
        else:
            imgs_val.append(Image(img_id, img_path))

    trainset = ImageSet(imgs_train, HEIGHT, WIDTH, 1)
    valset = ImageSet(imgs_val, HEIGHT, WIDTH, 1)

    X_train, y_train = trainset.get_x_y()
    X_val, y_val = valset.get_x_y()

    model.fit(x=X_train, y=y_train, epochs=10, verbose=1, callbacks=[model_checkpoint],
              validation_data=(X_val, y_val), batch_size=32)
