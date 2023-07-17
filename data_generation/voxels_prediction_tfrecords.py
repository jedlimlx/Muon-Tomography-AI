import time
import tqdm
import tensorflow as tf
from data_generation import read_muons_data, read_voxels_data


root = r"C:\Users\jedli\Downloads\data"


def serialize_example(x1, y1):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """

    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature = {
        "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x1).numpy()])),
        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y1).numpy()]))
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
    for i in tqdm.trange(2315):
        start_time = time.time()
        x = read_muons_data(f"{root}/muons/run_{i}.csv")
        y = read_voxels_data(f"{root}/voxels/run_{i}.npy")
        tf.print(time.time() - start_time)

        yield serialize_example(x, y)


serialized_features_dataset = tf.data.Dataset.from_generator(
    data_generator, output_types=tf.string, output_shapes=()
)

filename = f'voxels_prediction.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
