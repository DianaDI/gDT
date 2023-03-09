from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import open3d as o3d
import json
import re

from init import separated_mode_class_nums, ROOT_DIR
from data.kitti_helpers import labels


def print_ds_stats(paths, train, test, val):
    n = len(paths)
    print(f"TRAIN SIZE: {len(train)} kitti files ({len(train) / n * 100}%)")
    print(f"VAL SIZE: {len(val)} kitti files ({len(val) / n * 100}%)")
    print(f"TEST SIZE: {len(test)} kitti files ({len(test) / n * 100}%)")


def train_val_test_split(paths, test_size=0.1, seed=42, verbose=True):
    train, test = train_test_split(paths, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=test_size, random_state=seed)
    if verbose:
        print_ds_stats(paths, train, test, val)
    return train, test, val


def get_train_val_test_split_from_file(split_path, all_files, data_root_path, test_size=0.1, seed=42, verbose=True):
    # consider val as test now
    file_paths = open(split_path, 'r').readlines()
    test = file_paths
    for i in range(len(test)):
        test[i] = data_root_path + test[i].strip().split("train/")[-1]

    rest = []
    # get the rest
    split_file_names = [j.split("static/")[-1] for j in test]
    for f in all_files:
        if f[-25:] not in split_file_names:
            rest.append(f)

    train, val = train_test_split(rest, test_size=test_size, random_state=seed)
    if verbose:
        print_ds_stats(all_files, train, test, val)
    return train, test, val


def most_frequent(arr):
    occurence_count = Counter(arr)
    return occurence_count.most_common(1)[0][0]


def compute_normals(pos):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    o3d.geometry.PointCloud.estimate_normals(pcd,
                                             search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                 radius=0.1,
                                                 max_nn=100))
    return np.array(pcd.normals)


def compute_eigenv(pos, k_n=100):
    # code from https://github.com/denabazazian/Edge_Extraction/
    pcd1 = PyntCloud(pd.DataFrame(data=np.array(pos), columns=["x", "y", "z"]))
    # find neighbors
    kdtree_id = pcd1.add_structure("kdtree")
    k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id)
    # calculate eigenvalues
    ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    e1 = pcd1.points['e3(' + str(k_n + 1) + ')'].values
    e2 = pcd1.points['e2(' + str(k_n + 1) + ')'].values
    e3 = pcd1.points['e1(' + str(k_n + 1) + ')'].values

    # add normalisation
    e1 = np.array((e1 - np.min(e1)) / np.ptp(e1))
    e2 = np.array((e2 - np.min(e2)) / np.ptp(e2))
    e3 = np.array((e3 - np.min(e3)) / np.ptp(e3))

    return np.column_stack((e1, e2, e3))


def get_ignore_labels(mode):
    path = f'{ROOT_DIR}/mode{mode}_num_classes{separated_mode_class_nums[mode]}_res_label_map.json'
    with open(path, "r") as read_content:
        current_lbl_mapping = json.load(read_content)
    current_lbl_mapping = dict((v, k) for k, v in current_lbl_mapping.items())
    ignore_labels = []
    for l in labels:
        if l.ignoreInEval:
            if l.name in current_lbl_mapping.keys():
                ignore_labels.append(int(current_lbl_mapping[l.name]))
    if mode == 1 or mode == 2:
        ignore_labels.append(max(list(current_lbl_mapping.values())))
    return ignore_labels
