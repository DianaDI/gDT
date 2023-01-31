import os.path as osp
from glob import glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from os.path import basename
from tqdm import tqdm
import json
import re
import os
import multiprocessing
from joblib import Parallel, delayed

from data.kitti_helpers import id2name, ground_label_ids, all_label_ids
from data.pcd_utils import read_fields, cut_with_trajectory
from data.utils import compute_normals, compute_eigenv


class KITTI360Dataset(Dataset):
    def __init__(self, root, files, num_classes, mode=0, split="train", cut_in=2, transform=None, pre_transform=None,
                 pre_filter=None, normals=False, eigenvalues=False, ground_points_dir=None, poses_dir=None, config=None):
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

        self.config = config
        self.poses_dir = poses_dir
        self.cut_in = cut_in
        self.split = split
        self.normals = normals
        self.eigenvalues = eigenvalues
        self.files = files
        self.root = root
        self.mode = mode
        self.n_classes = num_classes
        self.ground_points_root = ground_points_dir
        self.class_weights_dict = None
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.class_label_id_save_path = f'{self.current_dir}/mode{self.mode}_num_classes{self.n_classes}_res_label_map.json'
        with open(f"{self.current_dir}/class_label_counts.json", "r") as read_content:
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

            folder_name = re.split(r'/|\\', raw_path)[-3]
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
        return osp.join(self.root, f'processed_mode_{self.mode}_traj_num_classes_{self.n_classes}_{self.config.data_suffix}')  # _h_dense')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
        return data


HIGHWAY_SCENES_FILES = ['2013_05_28_drive_0009_sync/static/0000013701_0000013838.ply',
                        '2013_05_28_drive_0009_sync/static/0000013575_0000013709.ply',
                        '2013_05_28_drive_0003_sync/static/0000000002_0000000282.ply',
                        '2013_05_28_drive_0003_sync/static/0000000274_0000000401.ply',
                        '2013_05_28_drive_0003_sync/static/0000000394_0000000514.ply',
                        '2013_05_28_drive_0003_sync/static/0000000508_0000000623.ply',
                        '2013_05_28_drive_0003_sync/static/0000000617_0000000738.ply',
                        '2013_05_28_drive_0003_sync/static/0000000731_0000000893.ply',
                        '2013_05_28_drive_0003_sync/static/0000000886_0000001009.ply',
                        '2013_05_28_drive_0007_sync/static/0000000002_0000000125.ply',
                        '2013_05_28_drive_0007_sync/static/0000000119_0000000213.ply',
                        '2013_05_28_drive_0007_sync/static/0000000208_0000000298.ply',
                        '2013_05_28_drive_0007_sync/static/0000000293_0000000383.ply',
                        '2013_05_28_drive_0007_sync/static/0000000378_0000000466.ply',
                        '2013_05_28_drive_0007_sync/static/0000000461_0000000547.ply',
                        '2013_05_28_drive_0007_sync/static/0000000542_0000000629.ply',
                        '2013_05_28_drive_0007_sync/static/0000000624_0000000710.ply',
                        '2013_05_28_drive_0007_sync/static/0000000705_0000000790.ply',
                        '2013_05_28_drive_0007_sync/static/0000000785_0000000870.ply',
                        '2013_05_28_drive_0007_sync/static/0000000865_0000000952.ply',
                        '2013_05_28_drive_0007_sync/static/0000000947_0000001039.ply',
                        '2013_05_28_drive_0007_sync/static/0000001034_0000001127.ply',
                        '2013_05_28_drive_0007_sync/static/0000001122_0000001227.ply',
                        '2013_05_28_drive_0007_sync/static/0000001221_0000001348.ply',
                        '2013_05_28_drive_0007_sync/static/0000001340_0000001490.ply',
                        '2013_05_28_drive_0007_sync/static/0000001483_0000001582.ply',
                        '2013_05_28_drive_0007_sync/static/0000001577_0000001664.ply',
                        '2013_05_28_drive_0007_sync/static/0000001659_0000001750.ply',
                        '2013_05_28_drive_0007_sync/static/0000001745_0000001847.ply',
                        '2013_05_28_drive_0007_sync/static/0000001841_0000001957.ply',
                        '2013_05_28_drive_0007_sync/static/0000001950_0000002251.ply',
                        '2013_05_28_drive_0007_sync/static/0000002237_0000002410.ply',
                        '2013_05_28_drive_0007_sync/static/0000002395_0000002789.ply',
                        '2013_05_28_drive_0007_sync/static/0000002782_0000002902.ply',
                        '2013_05_28_drive_0010_sync/static/0000000002_0000000208.ply',
                        '2013_05_28_drive_0010_sync/static/0000000199_0000000361.ply',
                        '2013_05_28_drive_0010_sync/static/0000000353_0000000557.ply',
                        '2013_05_28_drive_0010_sync/static/0000000549_0000000726.ply',
                        '2013_05_28_drive_0010_sync/static/0000000718_0000000881.ply',
                        '2013_05_28_drive_0010_sync/static/0000000854_0000000991.ply',
                        '2013_05_28_drive_0010_sync/static/0000000984_0000001116.ply',
                        '2013_05_28_drive_0010_sync/static/0000001109_0000001252.ply',
                        '2013_05_28_drive_0010_sync/static/0000001245_0000001578.ply',
                        '2013_05_28_drive_0010_sync/static/0000001563_0000001733.ply',
                        '2013_05_28_drive_0010_sync/static/0000001724_0000001879.ply',
                        '2013_05_28_drive_0010_sync/static/0000001872_0000002033.ply',
                        '2013_05_28_drive_0010_sync/static/0000002024_0000002177.ply',
                        '2013_05_28_drive_0010_sync/static/0000002168_0000002765.ply']
