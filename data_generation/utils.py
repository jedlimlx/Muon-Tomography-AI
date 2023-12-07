import numpy as np
import pandas as pd
import tensorflow as tf


def read_voxels_data(file):
    return np.load(file)


def read_muons_data(file):
    df = pd.read_csv(file)
    return df.to_numpy()


def read_trajectory_data(file, num=63):
    df = pd.read_csv(file, header=None)

    arr = df.to_numpy()

    muons = [[]]
    prev = df[0][0]
    for i in range(len(df)):
        row = arr[i]

        if row[0] == prev:
            if len(muons[-1]) == 0:
                muons[-1].append([row[1] / 1000 + 0.5, row[2] / 1000 + 0.5, row[3] / 1000 + 0.5])

            muons[-1].append([
                row[7] / 1000 + 0.5, row[8] / 1000 + 0.5, row[9] / 1000 + 0.5
            ])
        else:
            if len(muons[-1]) > 0 and muons[-1][-1][2] > -0.45:
                muons.pop()

            prev = row[0]
            muons.append([])

    if len(muons[-1]) == 0 or muons[-1][-1][2] > -0.45:
        muons.pop()

    muons = [x for x in muons if len(x) > 3]

    pts = np.arange(0, num+1) / num

    lst = []
    for muon in muons:
        lst.append(
            np.concatenate(
                [
                    np.interp(pts, [x[2] for x in muon][::-1], [x[0] for x in muon][::-1])[..., np.newaxis],
                    np.interp(pts, [x[2] for x in muon][::-1], [x[1] for x in muon][::-1])[..., np.newaxis]
                ], axis=-1
            )
        )

    return np.array(lst)


def predict_trajectory(x0, xf, p0, pf, z0=1.0, zf=0.0, num=63):
    x1 = x0
    x2 = xf - x0
    x3 = -(p0 - pf) * (z0 - zf)/(2 * np.pi)
    x4 = -(2 * (xf - x0) + (p0 + pf) * (z0 - zf)) / (4 * np.pi)

    pi = tf.cast(np.pi, tf.float32)
    z = tf.range(0, num+1, dtype=tf.float32)[::-1] / num
    return x1 + x2 * (z - z0) / (zf - z0) + \
        x3 * tf.sin((z - z0) * pi / (zf - z0)) + x4 * tf.sin(2 * (z - z0) * pi / (zf - z0))


if __name__ == "__main__":
    root = r"D:\muons_data\muons_trajectory"
    muons = read_trajectory_data(f"{root}/output/run_0.csv")
