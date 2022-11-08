from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import open3d as o3d


def train_val_test_split(paths, test_size=0.1, seed=42, verbose=True):
    train, test = train_test_split(paths, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=test_size, random_state=seed)
    if verbose:
        n = len(paths)
        print(f"TRAIN SIZE: {len(train)} kitti files ({len(train) / n * 100}%)")
        print(f"VAL SIZE: {len(val)} kitti files ({len(val) / n * 100}%)")
        print(f"TEST SIZE: {len(test)} kitti files ({len(test) / n * 100}%)")
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
                                                 max_nn=30))
    return np.array(pcd.normals)


def compute_eigenv(pos, k_n=30):
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
    return np.column_stack((e1, e2, e3))
