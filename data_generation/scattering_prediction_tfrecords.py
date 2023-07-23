import time
import tqdm
import tensorflow as tf
from data_generation import read_trajectory_data, read_voxels_data


root = r"C:\Users\jedli\Documents\muons-simulation"


def serialize_example(x, y, voxels):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
        "voxels": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(voxels).numpy()]))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tf_serialize_example(x, y):
    tf_string = tf.py_function(
        serialize_example,
        (x, y),
        tf.string
    )
    return tf.reshape(tf_string, ())


def data_generator():
    for i in tqdm.trange(2772):
        try:
            x, y = read_trajectory_data(f"{root}/output/run_{i}.csv")
            voxels = read_voxels_data(f"{root}/voxels/run_{i}.npy")

            yield serialize_example(x, y, voxels)
        except Exception as e:
            print(e)


serialized_features_dataset = tf.data.Dataset.from_generator(
    data_generator, output_types=tf.string, output_shapes=()
)

filename = f'../scattering_prediction.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
