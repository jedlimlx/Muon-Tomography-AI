import os
import tqdm
import threading

import numpy as np
import pandas as pd

SCALE = 2
RESOLUTION = 64

SENSOR_DIST = 10

DIRECTORY = r"D:\Muons Data"


def process(df, dose, write_path, name):
    run = False
    for i in range(20):
        if not os.path.exists(f"{write_path}/{name[:-4]}.npy"):
            run = True

    if not run: return

    lst = []

    df = df[[x < dose for x in range(len(df))]]
    bins = np.arange(-12 * SCALE, 12 * SCALE + 12 * SCALE / RESOLUTION, 24 * SCALE / RESOLUTION)

    # Calculating output detector planes
    output_df = df[(df["particleID"] == 13) & (df["x"] > 150 * SCALE)][
        ["y", "z", "px", "py", "pz", "count"]
    ]

    # Compute 1st plane
    output_df["y_cut"] = pd.cut(output_df["y"], bins=bins, right=False)
    output_df["z_cut"] = pd.cut(output_df["z"], bins=bins, right=False)
    pt = pd.pivot_table(output_df, columns="y_cut", index="z_cut", values="count", aggfunc="sum")
    lst.append(pt.values)

    # Compute ith plane
    for j in range(1, 20):
        t = SENSOR_DIST * j / df["px"]
        output_df["y2"] = output_df["y"] + output_df["py"] * t
        output_df["z2"] = output_df["z"] + output_df["pz"] * t

        min_y, max_y = np.min(output_df["y"].values), np.max(output_df["y"].values)
        min_z, max_z = np.min(output_df["z"].values), np.max(output_df["z"].values)
        df_2 = output_df[
            (min_y < output_df["y2"]) & (max_y > output_df["y2"]) & (min_z < output_df["z2"]) & (
                    max_z > output_df["y2"])
            ][["y2", "z2", "count"]]

        df_2["y2_cut"] = pd.cut(df_2["y2"], bins=bins, right=False)
        df_2["z2_cut"] = pd.cut(df_2["z2"], bins=bins, right=False)
        pt = pd.pivot_table(df_2, columns="y2_cut", index="z2_cut", values="count", aggfunc="sum")
        lst.append(pt.values)

    # Calculating input detector planes
    input_df = df[(df["particleID"] == 13) & (df["x"] > 150 * SCALE)][
        ["ver_y", "ver_z", "ver_px", "ver_py", "ver_pz", "count"]
    ]

    # Compute 1st plane
    input_df["y_cut"] = pd.cut(df["ver_y"], bins=bins, right=False)
    input_df["z_cut"] = pd.cut(df["ver_z"], bins=bins, right=False)
    pt = pd.pivot_table(input_df, columns="y_cut", index="z_cut", values="count", aggfunc="sum")
    lst.append(pt.values)

    # Compute ith plane
    for j in range(1, 20):
        t = -SENSOR_DIST * j / df["ver_px"]
        input_df["y2"] = input_df["ver_y"] + input_df["ver_py"] * t
        input_df["z2"] = input_df["ver_z"] + input_df["ver_pz"] * t

        min_y, max_y = np.min(input_df["ver_y"].values), np.max(input_df["ver_y"].values)
        min_z, max_z = np.min(input_df["ver_z"].values), np.max(input_df["ver_z"].values)
        df_2 = input_df[
            (min_y < input_df["y2"]) & (max_y > input_df["y2"]) & (min_z < input_df["z2"]) & (max_z > input_df["y2"])
            ][["y2", "z2", "count"]]

        df_2["y2_cut"] = pd.cut(df_2["y2"], bins=bins, right=False)
        df_2["z2_cut"] = pd.cut(df_2["z2"], bins=bins, right=False)
        pt = pd.pivot_table(df_2, columns="y2_cut", index="z2_cut", values="count", aggfunc="sum")
        lst.append(pt.values)

    lst = np.array(lst)
    np.save(f"{write_path}/{name[:-4]}.npy", lst)


def thread(num, dose):
    for csv in os.listdir(f"{DIRECTORY}/raw_detections"):
        if f"orient_{num}" in csv:
            print(csv)
            df = pd.read_csv(f"{DIRECTORY}/raw_detections/{csv}")
            process(df, dose, f"C:/Users/jedli/Documents/data/detections_{dose}", csv)


if __name__ == "__main__":
    for dose in [20000, 40000]:
        print(f"Running {dose} dose...")

        try: os.mkdir(f"C:/Users/jedli/Documents/data/detections_{dose}")
        except Exception as e: print(e)

        t1 = threading.Thread(target=thread, args=(0, dose))
        t2 = threading.Thread(target=thread, args=(1, dose))
        t3 = threading.Thread(target=thread, args=(2, dose))
        t4 = threading.Thread(target=thread, args=(3, dose))
        t5 = threading.Thread(target=thread, args=(4, dose))
        t6 = threading.Thread(target=thread, args=(5, dose))

        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()

        t1.join()
        t2.join()
        t3.join()
        t4.join()
        t5.join()
        t6.join()
