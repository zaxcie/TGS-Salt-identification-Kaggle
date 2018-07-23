from sklearn.model_selection import train_test_split
import os


def get_train_val_ids(path, train_size=0.8):
    img_ids = os.listdir("data/raw/train/images")

    train, val = train_test_split(img_ids, train_size=train_size, random_state=966)

    return train, val