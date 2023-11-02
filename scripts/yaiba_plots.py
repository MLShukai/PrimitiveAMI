import os

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font size
tick_fontsize = 24
label_fontsize = 24

# File and directory paths
data_dir = "../data/log.csv"
graph_dir = "../data/graph"

file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
exp_name = [os.path.splitext(f)[0] for f in file_list]

os.makedirs(os.path.join(graph_dir, "cycle_distance"), exist_ok=True)
os.makedirs(os.path.join(graph_dir, "angle"), exist_ok=True)
os.makedirs(os.path.join(graph_dir, "route"), exist_ok=True)
os.makedirs(os.path.join(graph_dir, "distance"), exist_ok=True)
os.makedirs(os.path.join(graph_dir, "CSV"), exist_ok=True)

for i, file_name in enumerate(file_list):
    input_file = os.path.join(data_dir, file_name)
    data = pd.read_csv(input_file, encoding="CP932", quotechar='"')

    data = data.iloc[:20000, :]

    # 1 second difference calculation
    data["x.diff"] = data["location_x"].diff().fillna(0) * -1
    data["z.diff"] = data["location_z"].diff().fillna(0) * -1

    data["ang.diff"] = data["rotation_2"].diff().fillna(0)
    data = data.dropna(subset=["ang.diff"])
    data.loc[data["ang.diff"] < -180, "ang.diff"] += 360
    data.loc[data["ang.diff"] > 180, "ang.diff"] -= 360
    data["ang.diff"] *= -1

    data["flag"] = np.logical_and(np.abs(data["x.diff"]) < 0.0001, np.abs(data["z.diff"]) < 0.0001)

    # Compute cycle number
    data["cycle"] = data["flag"].cumsum()

    data2 = data.loc[(~data["flag"]) & (~data["cycle"].isna())].copy()

    # Distance calculation
    data2["distance"] = np.sqrt(data2["x.diff"] ** 2 + data2["z.diff"] ** 2)

    # Compute distance size for each cycle
    data_min = data2.groupby("cycle").min()[["x.diff", "z.diff"]]
    data_max = data2.groupby("cycle").max()[["x.diff", "z.diff"]]

    data_diff = pd.merge(data_min, data_max, on="cycle")
    data_diff["siz"] = np.sqrt(
        (data_diff["x.diff_y"] - data_diff["x.diff_x"]) ** 2 + (data_diff["z.diff_y"] - data_diff["z.diff_x"]) ** 2
    )

    plt.figure(figsize=(12, 12))
    # plt.plot(data_diff.index, data_diff["siz"], "-o")
    plt.scatter(data_diff.index, data_diff["siz"])
    plt.xlabel("cycle.index")
    plt.ylabel("移動サイズ")
    plt.savefig(os.path.join(graph_dir, "cycle_distance", f"{exp_name[i]}.png"))
    plt.close()

    plt.figure(figsize=(12, 12))
    # plt.plot(data2.index, data2["ang.diff"], "-o")
    plt.scatter(data2.index, data2["ang.diff"], facecolor="None", edgecolors="black")
    plt.xlabel("経過時間（秒）", fontsize=label_fontsize)
    plt.ylabel("角度変化（度）", fontsize=label_fontsize)
    plt.xlim(0, 20000)
    plt.tick_params(labelsize=tick_fontsize)
    plt.savefig(os.path.join(graph_dir, "angle", f"{exp_name[i]}.png"))
    plt.close()

    plt.figure(figsize=(12, 12))
    plt.plot(data2["location_x"], data2["location_z"])
    plt.xlabel("x座標")
    plt.ylabel("z座標")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.savefig(os.path.join(graph_dir, "route", f"{exp_name[i]}.png"))
    plt.close()

    plt.figure(figsize=(12, 12))
    # plt.plot(data2.index, data2["distance"], "-o")
    plt.scatter(data2.index, data2["distance"])
    plt.xlabel("経過時間（秒）")
    plt.ylabel("移動距離")
    plt.xlim(0, 20000)
    plt.ylim(0, 2.0)
    plt.savefig(os.path.join(graph_dir, "distance", f"{exp_name[i]}.png"))
    plt.close()

    data2.to_csv(os.path.join(graph_dir, "CSV", f"{exp_name[i]}.csv"), index=False)
