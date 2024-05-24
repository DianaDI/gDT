import os
import random

from data.kitti_helpers import ground_label_ids

TASK_NAME = 'SemSegmentation'  # options: 'SemSegmentation'  # 'GroundDetection' # 'ObjectwiseSemSeg'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
linux = False
TRAIN_PATH = "path in linux machine" if linux else "local path"  # for older experiments it is required to specify path here instead of below
GROUND_SEP_ROOT = "path in linux machine" if linux else "local path"
POSES_DIR = "path in linux machine/" if linux else "local path"

mode = 1

COMMON_PARAMS = {
    'data_path': "your path to data folder",
    'val': False,
    'test': True,
    'data_suffix': "non_hw_only_50k_10_no_z_rot",
    'highway_files': False,
    'non_highway_files': False,
    'normalise': True,
    'random_seed': 402,
    'num_workers': 0,  # set number of cpu cores for data processing
    'plot_sample': True,
    'test_size': 0.1,
    'save_every': 10,
    'verbose': True,
    'normals': False,
    'eigenvalues': False,
    'resume_from': 0,
    'resume_from_id': 0,
    'resume_model_path': "your path to pretrained .pth model",
    'use_val_list': False,  # if false, random split with seed is used
    'val_list_path': "YOUR PATH/DATA/Kitti360/kitti_for_dva/kitti360mm/raw/data_3d_semantics/2013_05_28_drive_val_reduced.txt"
}

separated_mode_class_nums = {0: 37,
                             1: len(ground_label_ids) + 1,
                             2: 33}

MODEL_SPECIFIC_PARAMS = {
    'GroundDetection': {
        'lr': 0.001,
        'lr_decay': 0.99,  # every epoch
        'lr_cosine_step': None,
        'batch_size': 3,
        'num_epochs': 0,
        'subsample_to': 50000,
        'cut_in': 10,
        'num_classes': 2,
        'rand_rotation_x': 0.15,
        'rand_rotation_y': 0.15,
        'rand_rotation_z': 0.15,
        'params_log_file': "params.json",
        'batch_norm': True,
        'loss_fn': 'nll'  # options: nll, focal
    },
    'SemSegmentation': {
        'lr': 0.01,
        'lr_decay': 0,
        'lr_cosine_step': 5,
        'mode': mode,  # 1, 2, 0
        'num_classes': separated_mode_class_nums[mode],
        'batch_size': 1,
        'num_epochs': 300,
        'subsample_to': 50000,
        'cut_in': 5,
        'rand_rotation_x': 0.01,
        'rand_rotation_y': 0.01,
        'rand_rotation_z': 0.01,
        'params_log_file': "params.json",
        'eval_clustering': False,
        'batch_norm': True,
        'loss_fn': 'focal',  # options: nll, focal
        'clustering_eps': 0.025,  # 0.014, for mode 2
        'clustering_min_points': 10,  # 4 for mode 2
        'ignore_labels': False,  # now works only with focal loss,
        'load_predictions': False
    },
    'ObjectwiseSemSeg': {
        'label': 12,  # 0, 6, 11/12 - for NH
        'lr': 0.001,
        'lr_decay': 0,
        'lr_cosine_step': 5,  # if 0 - no scheduler applied
        'num_classes': 2,
        'batch_size': 1,
        'num_epochs': 200,
        'subsample_to': 50000,
        'rand_rotation_x': 10,
        'rand_rotation_y': 10,
        'rand_rotation_z': 360,
        'params_log_file': "params.json",
        'batch_norm': True,
        'loss_fn': 'nll',  # options: nll, focal
    }
}
