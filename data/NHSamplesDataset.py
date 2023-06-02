import os.path as osp
from glob import glob
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

from data.utils import compute_normals, compute_eigenv


def get_files_by_label(root, label):
    # select by label
    pc_files = glob(f'{root}/*_label{label}_*.ply')
    # label_files = glob(f'{root}/*_label{label}_*.dmp')
    return pc_files


class NHSamplesDataset(Dataset):

    def __init__(self, root, files, split="train", transform=None, pre_transform=None,
                 pre_filter=None,
                 config=None):
        self.split = split
        self.root = root
        self.files = files
        self.config = config
        self.split = split
        self.normals = config.normals
        self.eigenvalues = config.eigenvalues
        self.data_sample_save_prefix = f"{self.split}_data_"

        super().__init__(root, transform, pre_transform, pre_filter)
        self.transform = transform

    @property
    def raw_file_names(self):
        return self.files

    @property
    def processed_file_names(self):
        return glob(self.processed_dir + f'/{self.data_sample_save_prefix}*.pt')

    def label2binary(self, arr, label):
        return np.asarray([1 if i == label else 0 for i in arr])

    def proc_sample(self, pc_file, label_file, idx):
        pcd = o3d.io.read_point_cloud(pc_file)
        XYZ = torch.from_numpy(np.asarray(pcd.points))
        RGB = torch.from_numpy(np.asarray(pcd.colors) / 255)
        label = torch.from_numpy(self.label2binary(arr=np.load(label_file, allow_pickle=True), label=self.config.label))

        data = Data(pos=XYZ, x=RGB, y=label)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        if self.normals:
            normals = compute_normals(data.pos)
            data.x = torch.from_numpy(np.column_stack((data.x, normals)))

        if self.eigenvalues:
            eigenv = compute_eigenv(data.pos, k_n=500)
            data.x = torch.from_numpy(np.column_stack((data.x, eigenv)))

        torch.save(data, osp.join(self.processed_dir, f'{self.data_sample_save_prefix}{idx}.pt'))

    def process(self):

        # get label files based on pc files
        label_files = []
        for f in self.files:
            stem = Path(f).stem
            label_f = glob(f'{self.root}/*{stem}.dmp')[0]
            label_files.append(label_f)

        idx = 0
        for pc_file, l_file in tqdm(zip(self.files, label_files)):
            self.proc_sample(pc_file, l_file, idx)
            idx += 1

    def download(self):
        # Download to `self.raw_dir`.
        pass

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f'{self.config.label}_processed')

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.data_sample_save_prefix}{idx}.pt'))
        return data
