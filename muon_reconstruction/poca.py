import math
import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

import skimage.measure

import matplotlib
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
def poca(inputs_all, outputs_all, resolution=8):
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxels_2 = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    n_voxels = np.ones((resolution, resolution, resolution), dtype=np.float32)
    n_voxels = 2 * n_voxels

    for rotation in range(len(inputs_all)):
        inputs = inputs_all[rotation]
        outputs = outputs_all[rotation]

        # (x, y, z)
        # (-x, y, -z)
        # (-z, y, x)
        # (z, y, -x)
        # (-y, x, z)
        # (y, -x, z)
        def get_pixel(x):
            def subroutine(x):
                point = np.floor((x - np.array([335, -15, -15])) / 30 * resolution).astype(np.int32)
                # point[2] = resolution - point[2] - 1
                if rotation == 0: return point[0], point[1], point[2]
                elif rotation == 1: return resolution - point[0] - 1, point[1], resolution - point[2] - 1
                elif rotation == 2: return resolution - point[2] - 1, point[1], point[0]
                elif rotation == 3: return point[2], point[1], resolution - point[0] - 1
                elif rotation == 4: return resolution - point[1] - 1, point[0], point[2]
                elif rotation == 5: return point[1], resolution - point[0] - 1, point[2]

            return subroutine(x)

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
                pixel = get_pixel(point)
                voxels[pixel] += angle_scattered
                voxels_2[pixel] += angle_scattered ** 2

                # adding to voxels between initial point and scattering point
                a0 = a0 + a_hat * (335 / a_hat[0])
                grad = (point - a0) / (point[0] - a0[0])
                for t in range(int((point[0] - a0[0]))):
                    try: n_voxels[tuple(get_pixel(a0 + grad * t))] += 1
                    except IndexError: break

                # adding to voxels between scattering point and final point
                b0 = b0 - b_hat * ((450 - 365) / b_hat[0])
                grad = (b0 - point) / (b0[0] - point[0])
                for t in range(int(b0[0] - point[0])):
                    try: n_voxels[tuple(get_pixel(point + grad * t))] += 1
                    except IndexError: break

    return voxels_2 / n_voxels / (3 / resolution) * 250 ** 2 / 15


if __name__ == "__main__":
    dosages = [38000] # + [100, 200, 500, 700, 1000, 2000, 5000, 7000, 10000, 20000, 40000]
    mses = []

    for dose in dosages:
        print(f"Running {dose}...")

        total_mse = 0
        total_ssim = 0
        total_psnr = 0

        runs = 128
        start = 128
        for i in tqdm.trange(start, start + runs):
            index = i

            inputs_all = []
            outputs_all = []
            for rotation in range(6):  # loading inputs
                df = pd.read_csv(fr"D:/Muons Data/raw_detections/run_{index}_orient_{rotation}.csv")
                inputs = [[df["ver_y"][x], df["ver_z"][x], df["ver_py"][x] / df["ver_px"][x],
                           df["ver_pz"][x] / df["ver_px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10][:dose]
                outputs = [[df["y"][x], df["z"][x], df["py"][x] / df["px"][x],
                            df["pz"][x] / df["px"][x]] for x in range(len(df)) if df["ver_x"][x] < 10][:dose]

                inputs = np.array(inputs)
                outputs = np.array(outputs)

                inputs_all.append(inputs)
                outputs_all.append(outputs)

            inputs_all = np.array(inputs_all)
            outputs_all = np.array(outputs_all)

            # running algorithm
            result = poca(inputs_all, outputs_all)

            # loading ground truth
            voxels = np.load(f"D:/Muons Data/voxels/run_{index}.npy")
            voxels = skimage.measure.block_reduce(voxels, (8, 8, 8), np.max)

            # voxels = np.rot90(voxels, 2, axes=(0, 2))

            radiation_lengths = [0, 1 / 1.757, 1 / 1.424, 1 / 1.206, 1 / 0.8543, 1 / 0.5612, 1 / 0.3344, 1 / 0.3166]
            for k in range(1, len(radiation_lengths)):
                voxels[voxels == k] = radiation_lengths[k]

            # computing mse
            mse = np.average((result - voxels) ** 2)
            total_mse += mse

            # computing psnr
            # print(np.expand_dims(voxels, axis=-1).shape)
            psnr = tf.image.psnr(np.expand_dims(voxels, axis=-1), np.expand_dims(result, axis=-1), 3)
            total_psnr += psnr

            # computing ssim
            # ssim = tf.image.ssim(tf.constant(voxels.astype(np.int32)), tf.constant(result.astype(np.int32)), 3)
            # total_ssim += ssim

            # print(f"MSE: {total_mse / (i - start + 1)}")
            # print(f"PSNR: {total_psnr / (i - start + 1)}")
            # print(f"SSIM: {total_ssim / (i - start + 1)}")

        print(f"MSE: {total_mse / runs}")

        mses.append(total_mse / runs)
        # print(f"PSNR: {total_psnr / runs}")
        # print(f"SSIM: {total_ssim / runs}")

    """
    # plotting graphs
    plt.imshow(result[:, 2, :])
    plt.colorbar()
    plt.show()

    plt.imshow(voxels[:, 2, :])
    plt.colorbar()
    plt.show()

    colours = np.expand_dims(result, axis=-1)
    colours = np.repeat(np.sqrt(colours), 3, axis=-1)
    colours = 1 - colours / np.max(np.sqrt(result))
    colours[:, :, :, 2] = 1
    plot_voxels(result > 1e-1, colours)

    colours = np.expand_dims(voxels, axis=-1)
    colours = np.repeat(colours, 3, axis=-1)
    colours = 1 - colours / np.max(voxels)
    colours[:, :, :, 2] = 1
    plot_voxels(voxels > 0, colours)
    """
