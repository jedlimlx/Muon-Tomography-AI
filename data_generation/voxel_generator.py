import os
import cv2
import numpy as np
import tensorflow as tf

# generates hollow shapes with a high-Z object (cube) inside (nuclear non-proliferation)
def high_z_detection(
        resolution=64,
        shielding_mat=2,
        shielding_thickness=1/32,
        high_z_num=5,
        high_z_mat=10,
        high_z_radius=0.05
):
    # construct shielding
    x_min = int(np.random.uniform(low=0, high=0.1) * resolution)
    y_min = int(np.random.uniform(low=0, high=0.1) * resolution)
    z_min = int(np.random.uniform(low=0, high=0.1) * resolution)

    x_max = int((1 - np.random.uniform(low=0, high=0.1)) * resolution)
    y_max = int((1 - np.random.uniform(low=0, high=0.1)) * resolution)
    z_max = int((1 - np.random.uniform(low=0, high=0.1)) * resolution)

    voxels = np.zeros((resolution, resolution, resolution), dtype=np.int32)
    voxels[x_min:x_max, y_min:y_max, z_min:z_max] = shielding_mat

    t = int(shielding_thickness*resolution)
    voxels[x_min+t:x_max-t, y_min+t:y_max-t, z_min+t:z_max-t] = 0

    # construct high z object
    l = int(high_z_radius*resolution)
    for i in range(high_z_num):
        obj_x = int(np.random.uniform(low=x_min+t+l, high=x_max-t-l))
        obj_y = int(np.random.uniform(low=y_min+t+l, high=y_max-t-l))
        obj_z = int(np.random.uniform(low=z_min+t+l, high=z_max-t-l))

        voxels[obj_x-l:obj_x+l, obj_y-l:obj_y+l, obj_z-l:obj_z+l] = high_z_mat

    return voxels


# generate hollow spaces within a fully filled object (archaelogy)
def hidden_chamber(bg_mat=1, num_holes=5, resolution=64):
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.int32) + bg_mat

    pad = 5

    # construct high z object
    for i in range(num_holes):
        obj_x = int(np.random.uniform(low=pad, high=resolution-2*pad))
        obj_y = int(np.random.uniform(low=pad, high=resolution-2*pad))
        obj_z = int(np.random.uniform(low=pad, high=resolution-2*pad))

        volume = np.random.uniform(4, (resolution / 4) ** 3)
        w = int(np.cbrt(volume) * 2 ** np.random.normal(loc=0, scale=0.5))
        h = int(volume / np.cbrt(volume) / w * 2 ** np.random.normal(loc=0, scale=0.5))
        b = int(volume / w / h)

        voxels[obj_x:obj_x+w, obj_y:obj_y+h, obj_z:obj_z+b] = 0

    return voxels


def read_crosssections(path):
    lst = os.listdir(path)
    lst = [(int(x[-6:-4]), path + "/" + x) for x in lst]
    lst = sorted(lst, key=lambda x: x[0])

    voxels = []
    for img in lst:
        voxels.append(tf.image.resize(tf.io.decode_image(tf.io.read_file(img[1]))[..., 0:1], (64, 64)).numpy()[..., 0])

    return np.array(voxels)


if __name__ == "__main__":
    import os
    import random
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # voxels = read_crosssections("../crosssections")

    i = 0
    while i < 2400:
        try:
            np.save(
                f"../to_simulate/run_{i}.npy",

                hidden_chamber(
                    bg_mat=random.randint(4, 6),
                    num_holes=random.randint(1, 5),
                    resolution=64
                )
            )
        except ZeroDivisionError:
            continue

        i += 1

    """
    high_z_detection(
        shielding_mat=random.randint(1, 4),
        shielding_thickness=random.randint(2, 5)/64,
        high_z_mat=random.randint(9, 10),
        high_z_num=random.randint(0, 5),
        high_z_radius=random.uniform(0.03, 0.08)
    )
    """