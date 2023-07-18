import numpy as np
import pandas as pd


def read_voxels_data(file):
    return np.load(file)


def read_muons_data(file):
    df = pd.read_csv(file)
    return df.to_numpy()


def read_trajectory_data(file, k=5):
    df = pd.read_csv(file, header=None)

    arr = df.to_numpy()

    muons = [[]]
    prev = df[0][0]
    prev_direction = None
    for i in range(len(df)):
        row = arr[i]

        if row[0] == prev:
            if len(muons[-1]) == 0:
                prev_direction = np.array([row[4], row[5], row[6]])
                muons[-1].append([row[1] / 1000 + 0.5, row[2] / 1000 + 0.5, row[3] / 1000 + 0.5, row[4], row[5], row[6], 0])

            curr_direction = np.array([row[10], row[11], row[12]])
            if row[9] < -870 or (np.sum(np.square(prev_direction - curr_direction)) > 1e-5 and row[13] != 0):
                muons[-1].append([
                    row[7] / 1000 + 0.5, row[8] / 1000 + 0.5, row[9] / 1000 + 0.5,
                    curr_direction[0], curr_direction[1], curr_direction[2],
                    np.sum(np.square(prev_direction - curr_direction))
                    # row[10], row[11], row[12],
                    # row[13] % 64, row[13] % 64 // 64, row[13] // (64 * 64)
                ])
        else:
            prev = row[0]
            muons.append([])

    # filter out trajectories that are too short
    inputs = [np.array(x[0][:-1] + x[-1][:-1]) for x in muons if len(x) >= 2]
    
    def f(x):

        if len(x) == 2:
            return np.zeros((1+k*3,))

        x = np.array(x[1:-1])[:, [0, 1, 2, 6]]

        if len(x) < k:
            temp = np.concatenate([x, np.zeros((k-len(x), 4))], axis=0)
        else:
            temp = x

        return np.concatenate([[min(len(x), 5)], temp[np.sort(np.argpartition(temp[:, 3], -k)[-k:]), :3].flatten()], axis=0)
    
    target = [f(x) for x in muons if len(x) >= 2]
    return np.array(inputs), np.array(target)


if __name__ == "__main__":
    root = r"C:\Users\jedli\Downloads\data"
    x, y = read_trajectory_data(f"{root}/output/run_0.csv")
