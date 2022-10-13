import glob
import numpy as np
from plyfile import PlyData
import open3d as o3d
from tqdm import tqdm
import json

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

# XYZ, RGB, labels = read_fields("C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/2013_05_28_drive_0000_sync/static/0000005880_0000006165.ply")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(XYZ)
# pcd.colors = o3d.utility.Vector3dVector(RGB)
# o3d.visualization.draw_geometries([pcd])


# all_files = glob.glob("C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/*_sync/static/*.ply")
# counts = {}
# for file in tqdm(all_files):
#     _, _, labels = read_fields(file, xyz=False, rgb=False, label=True)
#     for l in labels:
#         l = int(l)
#         if l in counts.keys():
#             counts[l] += 1
#         else:
#             counts[l] = 1
# print(counts)
# with open("class_label_counts.json", 'w') as fp:
#     json.dump(counts, fp)
