import os.path as osp
from glob import glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from os.path import basename
from tqdm import tqdm
import json
import multiprocessing
from joblib import Parallel, delayed

from data.kitti_helpers import label_names, id2name, ground_label_ids, all_label_ids
from data.pcd_utils import read_fields, cut_with_trajectory
from data.utils import compute_normals, compute_eigenv


class KITTI360Dataset(Dataset):
    def __init__(self, root, files, num_classes, mode=0, split="train", cut_in=2, transform=None, pre_transform=None,
                 pre_filter=None, normals=False, eigenvalues=False):
        """

        :param root:
        :param files:
        :param mode: 0 - general training, 1 - only on ground points, 2 - only on non-ground points
        :param split:
        :param cut_in:
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        self.poses_dir = "C:/Users/Diana/Desktop/DATA/Kitti360/data_poses"
        self.cut_in = cut_in
        self.split = split
        self.normals = normals
        self.eigenvalues = eigenvalues
        self.files = files
        self.root = root
        self.mode = mode
        self.num_classes = num_classes
        self.ground_points_root = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed/inliers_traj_0.6"
        self.class_weights_dict = None
        self.class_label_id_save_path = f'./mode{self.mode}_num_classes{self.num_classes}_res_label_map.json'
        with open("./class_label_counts.json", "r") as read_content:
            self.class_weights_dict_original = json.load(read_content)
            self.class_weights_dict_original = dict(
                [(int(k), self.class_weights_dict_original[k]) for k in self.class_weights_dict_original])

        self.non_ground_ids = list(
            set(all_label_ids).intersection(list(self.class_weights_dict_original)) - set(ground_label_ids))
        if self.mode == 2:
            self.mapping_labels_ids = self.non_ground_ids
        elif self.mode == 1:
            self.mapping_labels_ids = ground_label_ids
        elif self.mode == 0:
            self.mapping_labels_ids = sorted(list(self.class_weights_dict_original.keys()))
        self.res_mapping, self.label_mapping, other_label = self.get_new_mapping(self.mapping_labels_ids, id2name,
                                                                                 self.mode)
        new_class_weights_dict = dict()
        for label in self.label_mapping:
            new_class_weights_dict[self.label_mapping[label]] = self.class_weights_dict_original[label]
        self.class_weights_dict = new_class_weights_dict
        if self.mode != 0:
            self.class_weights_dict[other_label] = 0
            for label in ground_label_ids:
                self.class_weights_dict[other_label] += self.class_weights_dict_original[label]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.transform = transform
        self.seg_classes = id2name

    @property
    def raw_file_names(self):
        # return glob(self.root + "\\*\\static\\*.ply")
        return self.files

    @property
    def processed_file_names(self):
        return glob(self.processed_dir + f'/{self.split}_data_*.pt')

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def get_new_mapping(self, label_ids, id_name_dict, mode):
        res_mapping = {}
        label_mapping = dict(zip(label_ids, range(0, len(label_ids))))
        for key, val in label_mapping.items():
            res_mapping[val] = id_name_dict[key]  # 1: "name"
        # all other classes are mapped to one label

        other_label = len(label_ids)
        if mode != 0:
            res_mapping[other_label] = "other"
        return res_mapping, label_mapping, other_label

    def map_labels(self, main_arr, label_ids, id_name_dict, mode, save_label_map=False,
                   file_path_to_save="./mapping.json"):
        res_mapping, label_mapping, other_label = self.get_new_mapping(label_ids, id_name_dict, mode)
        res_mapping, label_mapping = self.res_mapping, self.label_mapping
        for i in range(len(main_arr)):
            if main_arr[i][-1] in label_ids:
                main_arr[i][-1] = label_mapping[main_arr[i][-1]]
            else:
                if mode != 0:
                    main_arr[i][-1] = other_label
        if save_label_map:
            with open(file_path_to_save, 'w') as fp:
                json.dump(res_mapping, fp, indent=2)
        return main_arr

    def pre_process(self, mode, data, path):
        res = None
        if mode == 0:
            res = self.map_labels(data, self.mapping_labels_ids, id2name, self.mode, True,
                                  file_path_to_save=self.class_label_id_save_path)
        if mode == 1:
            # filter out ground points
            ground_points = np.load(f'{self.ground_points_root}/{basename(path).split(".")[0]}.pkl',
                                    allow_pickle=True)
            res = data[ground_points]
            res = self.map_labels(res, ground_label_ids, id2name, self.mode, True,
                                  file_path_to_save=self.class_label_id_save_path)
        elif mode == 2:
            # filter out non-ground points
            ground_points = np.load(f'{self.ground_points_root}/{basename(path).split(".")[0]}.pkl',
                                    allow_pickle=True)
            res = data[list(set(range(len(data))) - set(ground_points))]
            res = self.map_labels(res, self.non_ground_ids, id2name, self.mode, True,
                                  file_path_to_save=self.class_label_id_save_path)
        return res

    def proc_sample(self, part, idx):
        XYZ = torch.from_numpy(part[:, :3])
        RGB = torch.from_numpy(part[:, 3:6])
        label = torch.from_numpy(part[:, -1])

        data = Data(pos=XYZ, x=RGB, y=label)

        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        if self.normals:
            normals = compute_normals(data.pos)
            data.x = torch.from_numpy(np.column_stack((data.x, normals)))

        if self.eigenvalues:
            eigenv = compute_eigenv(data.pos, k_n=500)
            data.x = torch.from_numpy(np.column_stack((data.x, eigenv)))

        torch.save(data, osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))

    def process(self):
        # cut_volumes = []
        idx = 0
        for raw_path_idx in tqdm(range(len(self.raw_paths))):
            raw_path = self.raw_paths[raw_path_idx]
            XYZ, RGB, label = read_fields(raw_path)
            all = np.column_stack((XYZ, RGB, label))
            all = self.pre_process(self.mode, all, raw_path)

            folder_name = raw_path.split("\\")[1]
            trajectory_poses = open(f"{self.poses_dir}/{folder_name}/poses.txt", "r").read().splitlines()
            splits = cut_with_trajectory(n=self.cut_in, pcd_path=raw_path, traj_poses=trajectory_poses,
                                         xyz=all[:, :3], rgb=all[:, 3:6], labels=all[:, -1])

            splits_len = len(splits)
            # num_cores = multiprocessing.cpu_count() - 4
            # Parallel(n_jobs=num_cores)(
            #     delayed(self.proc_sample)(splits, i, splits_len, raw_path_idx) for i in range(splits_len))
            for i in range(splits_len):
                self.proc_sample(splits[i], idx)
                idx += 1

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_mode_{self.mode}_traj_num_classes_{self.num_classes}')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
        return data
