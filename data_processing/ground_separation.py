import open3d as o3d
from tqdm import tqdm
from os.path import basename
import numpy as np
import os
from plyfile import PlyData, PlyElement

from init import train_files, val_files
from data.pcd_utils import read_fields


def separate_ground_by_plane(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.25,
                                             ransac_n=3,
                                             num_iterations=5000)

    # xyz, rgb, label = read_fields(pcd_path)
    # all = np.column_stack((xyz, rgb, label))
    # inliers_arr = all[inliers]
    # outlier_arr = all[list(set(range(len(all))) - set(inliers))]
    return inliers


def prep_to_save(arr):
    res = np.array(arr,
                   dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                          ('semantic', 'f4')])
    return res


def process(f, mode):
    ground = np.asarray(separate_ground_by_plane(f))
    # save_path_ground = f'{save_dir}/{mode}/ground/{basename(f)}'
    # save_path_nonground = f'{save_dir}/{mode}/non-ground/{basename(f)}'

    save_dump_root = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed/inlier_indices"
    save_path = f'{save_dump_root}/{basename(f).split(".")[0]}.pkl'
    if not os.path.exists(save_path):
        ground.dump(save_path)
    else:
        print("repeated name!")

    # ground = prep_to_save(ground)
    # vertices = []
    # for i, row in enumerate(ground):
    #     el = PlyElement.describe(row, f'vertex{i}')
    #     vertices.append(el)
    # PlyData(vertices).write('./test.ply')
    # print("Hello")
    # o3d.io.write_point_cloud(save_path_ground, ground)
    # o3d.io.write_point_cloud(save_path_nonground, non_ground)


if __name__ == "__main__":

    save_dir = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed"

    for f in tqdm(train_files):
        process(f, "train")

    for f in tqdm(val_files):
        process(f, "val")
