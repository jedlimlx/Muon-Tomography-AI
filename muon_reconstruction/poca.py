import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_voxels(data, colours=None):
    ax = plt.figure().add_subplot(projection='3d')
    if colours is None:
        ax.voxels(data)
    else:
        ax.voxels(
            data,
            facecolors=colours,
            edgecolors=np.clip(2*colours - 0.5, 0, 1)
        )

    plt.show()


# Basically find the point of closest approach of the incoming and outgoing rays
# and denote that as were the muon scattered.
# inputs [y, z, py/px, pz/px]
def poca(inputs, outputs, resolution=64):
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float)
    voxels_2 = np.zeros((resolution, resolution, resolution), dtype=np.float)
    n_voxels = np.ones((resolution, resolution, resolution), dtype=np.float)
    n_voxels = 2 * n_voxels
    for i in range(len(inputs)):
        # define lines A and B by two points
        a0 = np.array([0, inputs[i, 0], inputs[i, 1]])
        a1 = np.array([-2, inputs[i, 0] - 2 * inputs[i, 2], inputs[i, 1] - 2 * inputs[i, 3]])
        b0 = np.array([450, outputs[i, 0], outputs[i, 1]])
        b1 = np.array([452, outputs[i, 0] + 2 * outputs[i, 2], outputs[i, 1] + 2 * outputs[i, 3]])

        # compute unit vectors of directions of lines A and B
        a_hat = (a1 - a0) / np.linalg.norm(a1 - a0)
        b_hat = (b1 - b0) / np.linalg.norm(b1 - b0)

        # find unit direction vector for line C, which is perpendicular to lines A and B
        c_hat = np.cross(b_hat, a_hat)
        c_hat /= np.linalg.norm(c_hat)

        # find angle scattered
        angle1 = math.atan(abs((a_hat[1] ** 2 + a_hat[2] ** 2) / a_hat[0]))
        angle2 = math.atan(abs((b_hat[1] ** 2 + b_hat[2] ** 2) / b_hat[0]))
        angle_scattered = abs(angle1 - angle2)

        # solve the system
        sol = np.linalg.solve(np.array([a_hat, -b_hat, c_hat]).T, b0 - a0)

        # add scattering density
        point = (a0 + sol[0] * a_hat + b0 + sol[1] * b_hat) / 2
        if 335 < point[0] < 365 and -15 < point[1] < 15 and -15 < point[2] < 15:
            pixel = np.floor((point - np.array([335, -15, -15])) / 30 * resolution).astype(np.int32)
            pixel = tuple(pixel)
            voxels[pixel] += angle_scattered
            voxels_2[pixel] += angle_scattered ** 2
            n_voxels[pixel] += 1

    result = 1 / (n_voxels - 1) * (voxels_2 - voxels ** 2 / n_voxels)
    return result


if __name__ == "__main__":
    df = pd.read_csv("run_12.csv")
    inputs = [[df["ver_y"][x], df["ver_z"][x], df["ver_py"][x] / df["ver_px"][x],
               df["ver_pz"][x] / df["ver_px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10]
    outputs = [[df["y"][x], df["z"][x], df["py"][x] / df["px"][x],
                df["pz"][x] / df["px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10]

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    result = poca(inputs, outputs)
    colours = np.expand_dims(np.sqrt(np.sqrt(result)), axis=-1)
    colours = np.repeat(colours, 3, axis=-1)
    colours = 1 - colours / np.max(np.sqrt(np.sqrt(result)))
    colours[:, :, :, 2] = 1
    plot_voxels(result > 2e-6, colours)

    voxels = np.load("run_12.npy")
    colours = np.expand_dims(voxels, axis=-1)
    colours = np.repeat(colours, 3, axis=-1)
    colours = 1 - colours / np.max(voxels)
    colours[:, :, :, 2] = 1
    plot_voxels(voxels > 3, colours)
