import subprocess
import threading
import numpy as np
import pandas as pd
import os
import sys

RESOLUTION = 64
N = 1000
N_threads = 6
FILE_PATH = sys.path[0]

def run_threads(run, i, voxels):
    bins = np.arange(-12, 12 + 12/RESOLUTION,24/RESOLUTION)
    np.savetxt("run_" + str(i) + ".voxel", voxels.flatten(), delimiter=" ")
    args = ["mu", "run.mac", "run_" + str(i)]
    proc = subprocess.Popen(args = [os.path.join(FILE_PATH, "mu"), "run.mac", "run_" + str(i)], stdout=subprocess.DEVNULL)
    proc.wait()
    print(i)
    txt_df = pd.read_csv("run_" + str(i) + ".txt", delimiter=" ", header=None, 
                                 names=["event", "count", "x", "y", "z", "time", "eIn", "eDep", 
                                          "trackID", "copyNo", "particleID"])
    txt_df = txt_df[(txt_df["particleID"] == 13) & (txt_df["x"] > 150)][["y", "z", "count"]]
    txt_df["y_cut"] = pd.cut(txt_df["y"], bins = bins, right = False)
    txt_df["z_cut"] = pd.cut(txt_df["z"], bins = bins, right = False)
    pt = pd.pivot_table(txt_df, columns = "y_cut", index = "z_cut", values="count", aggfunc="sum")
    np.save("detections/" + str(run) + "_orient_" + str(i) + ".npy", pt.values)


def rotate_cube(cuberay):
    res = []
    res.append(cuberay)
    res.append(np.rot90(cuberay, 2, axes=(0,2)))
    res.append(np.rot90(cuberay, axes=(0,2)))
    res.append(np.rot90(cuberay, -1, axes=(0,2)))
    res.append(np.rot90(cuberay, axes=(0,1)))
    res.append(np.rot90(cuberay, -1, axes=(0,1)))
    return res

def main():
    threads = []
    for j in range(N):
        noise = PerlinNoise(seed = j)
        voxels = np.array([[[noise.noise(x / RESOLUTION, y / RESOLUTION, z / RESOLUTION) > 0.6 for x in range(RESOLUTION)] for y in range(RESOLUTION)] for z in range(RESOLUTION)])
        orientations = rotate_cube(voxels)
        for i in range(N_threads):
            th = threading.Thread(target=run_threads, args=(j, i, orientations[i]))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        np.save("voxels/run_" + str(j) + ".npy", voxels)
        print("Run", j)

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
    
def lerp(t, a, b):
    return a + t * (b - a)

def grad(ahash, x, y, z):
    # very screwed up bitwise operations
    h = ahash & 15
    u = x if h < 8 else y
    v = y if h < 4 else x if (h == 12 or h == 14) else z
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

class PerlinNoise: # because the library has "undesired behaviour"

    def __init__(self, seed=None):
        if seed is None:
            p = np.arange(256)
            np.random.shuffle(p)
            self.p = np.tile(p, 2)
        else:
            np.random.seed(seed)
            p = np.arange(256)
            np.random.shuffle(p)
            self.p = np.tile(p, 2)
    
    def noise(self, x, y, z): # i have no idea what this does
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        Z = int(np.floor(z)) & 255

        x -= np.floor(x)
        y -= np.floor(y)
        z -= np.floor(z)
        
        u = fade(x)
        v = fade(y)
        w = fade(z)

        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z

        # ???
        res = lerp(w, lerp(v, lerp(u, grad(self.p[AA], x, y, z), grad(self.p[BA], x-1, y, z)), lerp(u, grad(self.p[AB], x, y-1, z), grad(self.p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(self.p[AA+1], x, y, z-1), grad(self.p[BA+1], x-1, y, z-1)), lerp(u, grad(self.p[AB+1], x, y-1, z-1),	grad(self.p[BB+1], x-1, y-1, z-1))))
        return (res + 1.0)/2.0


if __name__ == "__main__":
    main()