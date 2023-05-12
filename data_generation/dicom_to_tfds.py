import os
import pydicom

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def example(img):
    feature = {
        'img': _bytes_feature(tf.io.serialize_tensor(img)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

root = r"D:\CT Scan Data\Colorectal Cancer\manifest-1646429317311"
df = pd.read_csv(f"{root}/metadata.csv")

tfr = r"D:\colorectal.tfrecord"
with tf.io.TFRecordWriter(tfr) as writer:
    for file in tqdm(df["File Location"]):
        directory = f"{root}/{file[2:]}"

        for img in os.listdir(directory):
            if "xml" in img: continue

            ds = pydicom.dcmread(f"{root}/{file[2:]}/{img}")
            # if np.min(ds.pixel_array) != -1024: continue

            # pixels = ds.pixel_array #np.clip(, 0, None)
            pixels = np.clip(ds.pixel_array, 0, None) / 3071 * MU_MAX / 10

            pixels = tf.convert_to_tensor(pixels, dtype=tf.float32)
            tf_example = example(pixels)
            writer.write(tf_example.SerializeToString())

            """
            plt.imshow(pixels)
            plt.show()
    
            counts, bins = np.histogram(pixels, bins=100)
            plt.stairs(counts, bins)
            plt.show()
    
            print(np.mean(pixels))
            print(np.std(pixels))
            print(np.max(pixels))
            print(np.min(pixels))
            print(pixels.shape)
            print()
            """