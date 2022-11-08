import glob
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import json
from collections import defaultdict
from os.path import basename


def get_nearest_point(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def get_trajectory_points(pcd_path, poses):
    ranges = basename(pcd_path).split(".")[0].split("_")
    start, end = int(ranges[0]), int(ranges[1])
    points = []
    for line in poses:
        l_parts = line.split(" ")
        if start <= int(l_parts[0]) <= end:
            x, y, z = l_parts[4], l_parts[8], l_parts[-1]
            points.append(np.asarray((float(x), float(y), float(z))))
    return np.asarray(points)


def cut_with_trajectory(n, pcd_path, traj_poses, xyz, rgb, labels):
    traj_points = get_trajectory_points(pcd_path, traj_poses)
    part_len = int(len(traj_points) / n)
    split_traj_points = [traj_points[i] for i in range(part_len, len(traj_points), part_len)]

    parts_dict_xyz, parts_dict_rgb, parts_dict_lbl = defaultdict(list), defaultdict(list), defaultdict(list)

    for i in range(len(xyz)):
        p, r, l = xyz[i], rgb[i], labels[i]
        nearest = get_nearest_point(p, split_traj_points)
        parts_dict_xyz[nearest].append(p)
        parts_dict_rgb[nearest].append(r)
        parts_dict_lbl[nearest].append(l)

    split_parts = []

    for i in range(len(split_traj_points)):
        arr_part = np.column_stack((parts_dict_xyz[i], parts_dict_rgb[i], parts_dict_lbl[i]))
        split_parts.append(arr_part)

    return split_parts


def cut_boxes(arr, n):
    # decide whether to cut along x or y
    x_len = max(arr[:, 0]) - min(arr[:, 0])
    y_len = max(arr[:, 1]) - min(arr[:, 1])
    axis = 0 if x_len >= y_len else 1

    ax = arr[:, axis]
    part_len = (np.max(ax) - np.min(ax)) / n
    arrs = list()
    for i in range(n):
        if i != n - 1:
            mask = np.logical_and(arr[:, axis] > np.min(ax) + part_len * i,
                                  arr[:, axis] < np.min(ax) + part_len * (i + 1))
            arr_temp = arr[mask]
        else:
            arr_temp = arr[arr[:, axis] > np.min(ax) + part_len * i]
        arrs.append(arr_temp)
    return arrs


def read_fields(path, xyz=True, rgb=True, label=True):
    pcd = PlyData.read(path)
    pcdv = pcd['vertex']
    XYZ = np.column_stack((pcdv['x'], pcdv['y'], pcdv['z'])) if xyz else None
    RGB = np.column_stack((pcdv['red'] / 255, pcdv['green'] / 255, pcdv['blue'] / 255)) if rgb else None
    labels = np.array(pcdv['semantic']) if label else None
    return XYZ, RGB, labels


def count_class_labels():
    all_files = glob.glob("C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/*_sync/static/*.ply")
    counts = {}
    for file in tqdm(all_files):
        _, _, labels = read_fields(file, xyz=False, rgb=False, label=True)
        for l in labels:
            l = int(l)
            if l in counts.keys():
                counts[l] += 1
            else:
                counts[l] = 1
    print(counts)
    with open("class_label_counts.json", 'w') as fp:
        json.dump(counts, fp)
