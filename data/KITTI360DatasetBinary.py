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


class KITTI360DatasetBinary(Dataset):
    def __init__(self, root, files, split, cut_in=2, transform=None, pre_transform=None,
                 pre_filter=None):
        """

        :param root:
        :param files:
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
        self.num_classes = 2
        # 0 - ground
        # 1 - non-ground
        self.ground_ids = ground_label_ids

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

    def process(self):
        idx = 0
        cut_volumes = []
        for raw_path in tqdm(self.raw_paths):
            XYZ, RGB, label = read_fields(raw_path)

            for i in range(len(label)):
                label[i] = 0 if label[i] in self.ground_ids else 1

            all = np.column_stack((XYZ, RGB, label))
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

                torch.save(data, osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
                idx += 1
        print(f'Mean num of points in a cut before downsamplimg: {np.mean(cut_volumes)}')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'processed_binary')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.split}_data_{idx}.pt'))
        return data
