import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_voxels(data1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(data1)
    plt.show()


# Basically find the point of closest approach of the incoming and outgoing rays
# and denote that as were the muon scattered.
# inputs [y, z, py/px, pz/px]
def poca(inputs, outputs, size=(64, 64, 64)):
    voxels = []
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
        if angle_scattered > 0.01 and 320 < point[0] < 340:
            voxels.append(point)

    return voxels


if __name__ == "__main__":
    df = pd.read_csv("run_0.csv")
    inputs = [[df["ver_y"][x], df["ver_z"][x], df["ver_py"][x] / df["ver_px"][x],
               df["ver_pz"][x] / df["ver_px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10]
    outputs = [[df["y"][x], df["z"][x], df["py"][x] / df["px"][x],
                df["pz"][x] / df["px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10]

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    result = poca(inputs, outputs)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([x[0] for x in result], [x[1] for x in result], [x[2] for x in result])
    plt.show()

    voxels = np.load("run_0.npy")
    plot_voxels(voxels)
