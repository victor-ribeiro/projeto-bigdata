import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import utils

config = utils.get_config()


def load_tf_dataset(name: str, split) -> tuple:
    ds = tfds.load(name=name, split=split, shuffle_files=True)
    return ds


def resize_img(tensor):
    features = tensor["image"]
    label = tensor["label"]
    return tf.image.resize(features, config["TENSOR_SPEC"][:-1]), tf.one_hot(
        label, depth=3
    )


def normalize_img(image, label):
    return image / 255, label


def prepair_ds(tensor):
    resized_ds = tensor.map(map_func=resize_img)
    normalized_ds = resized_ds.map(normalize_img)
    dataset = normalized_ds.repeat().batch(config["BATCH_SIZE"])
    return dataset


def label_to_nparray(tensor, steps):
    label = []
    for i, (img, tmp_label) in enumerate(tfds.as_numpy(tensor)):
        if i > steps:
            break
        label.extend(tmp_label.tolist())
    return np.array(label)
