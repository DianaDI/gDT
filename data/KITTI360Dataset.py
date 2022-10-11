import os.path as osp
from glob import glob

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from plyfile import PlyData
from os.path import basename
from tqdm import tqdm
import open3d as o3d
import json

from data.kitti_helpers import label_names, id2name, ground_label_ids, all_label_ids
from data.pcd_utils import read_fields, cut_boxes


class KITTI360Dataset(Dataset):
    def __init__(self, root, files, num_classes, mode=0, split="train", cut_in=2, transform=None, pre_transform=None,
                 pre_filter=None):
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
        self.cut_in = cut_in
        self.split = split
        self.files = files
        self.root = root
        self.mode = mode
        self.num_classes = num_classes
        self.ground_points_root = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed/inliers_traj_0.6"
        self.class_weights_dict = None
        with open("./class_label_counts.json", "r") as read_content:
            self.class_weights_dict_original = json.load(read_content)
            self.class_weights_dict_original = dict([(int(k), self.class_weights_dict_original[k]) for k in self.class_weights_dict_original])

        self.non_ground_ids = list(set(all_label_ids).intersection(list(self.class_weights_dict_original)) - set(ground_label_ids))
        if self.mode == 2:
            self.mapping_labels_ids = self.non_ground_ids
        elif self.mode == 1:
            self.mapping_labels_ids = ground_label_ids
        elif self.mode == 0:
            self.mapping_labels_ids = sorted(list(self.class_weights_dict_original.keys()))
        self.res_mapping, self.label_mapping, other_label = self.get_new_mapping(self.mapping_labels_ids, id2name, self.mode)
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
            res_mapping[val] = id_name_dict[key]   # 1: "name"
        # all other classes are mapped to one label

        other_label = len(label_ids)
        if mode != 0:
            res_mapping[other_label] = "other"
        return res_mapping, label_mapping, other_label

    def map_labels(self, main_arr, label_ids, id_name_dict, mode, save_label_map=False, file_path_to_save="./mapping.json"):
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

    def process(self):
        idx = 0
        cut_volumes = []
        for raw_path in tqdm(self.raw_paths):
            XYZ, RGB, label = read_fields(raw_path)
            all = np.column_stack((XYZ, RGB, label))

            if self.mode == 0:
                all = self.map_labels(all, self.mapping_labels_ids, id2name, self.mode, True,
                                          file_path_to_save=f'./mode{self.mode}_num_classes{self.num_classes}_res_label_map.json')
            if self.mode == 1:
                # filter out ground points
                ground_points = np.load(f'{self.ground_points_root}/{basename(raw_path).split(".")[0]}.pkl',
                                     allow_pickle=True)
                all = all[ground_points]
                all = self.map_labels(all, ground_label_ids, id2name, self.mode, True,
                                          file_path_to_save=f'./mode{self.mode}_num_classes{self.num_classes}_res_label_map.json')
            elif self.mode == 2:
                # filter out non-ground points
                ground_points = np.load(f'{self.ground_points_root}/{basename(raw_path).split(".")[0]}.pkl',
                                     allow_pickle=True)
                all = all[list(set(range(len(all))) - set(ground_points))]
                all = self.map_labels(all, self.non_ground_ids, id2name, self.mode, True,
                                          file_path_to_save=f'./mode{self.mode}_num_classes{self.num_classes}_res_label_map.json')

            splits = cut_boxes(all, self.cut_in)

            for part in splits:
                cut_volumes.append(len(part))
                XYZ = part[:, :3]
                RGB = part[:, 3:6]
                label = part[:, -1]


                data = Data(pos=torch.from_numpy(XYZ), x=torch.from_numpy(RGB), y=torch.from_numpy(label))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data.x)

                o3d.geometry.PointCloud.estimate_normals(
                    pcd,
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                                      max_nn=30))

                data.x = torch.from_numpy(np.column_stack((data.x, np.array(pcd.normals))))

                torch.save(data, osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
                idx += 1
        print(f'Mean num of points in a cut before downsamplimg: {np.mean(cut_volumes)}')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_mode_{self.mode}_traj_num_classes_{self.num_classes}')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
        return data
