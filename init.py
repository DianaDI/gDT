import os
import random

from data.kitti_helpers import ground_label_ids

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
linux = False
TRAIN_PATH = "/rds/user/dd593/hpc-work/data_3d_semantics/train/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/kitti_for_dva/kitti360mm/raw/data_3d_semantics/"  # "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"
GROUND_SEP_ROOT = "/rds/user/dd593/hpc-work/inliers_traj_0.6/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed/inliers_traj_0.6"
POSES_DIR = "/rds/user/dd593/hpc-work/data_poses/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/data_poses"

# ROOT_PART_OBJECTWISE = "C:/Users/Diana/Desktop/DATA/NH_dataset/cut_objects6/"

random_id = random.randint(0, 1000)

mode = 0

COMMON_PARAMS = {
    'data_path': "C:/Users/Diana/Desktop/DATA/Kitti360/BBsv2/",
    # "C:/Users/Diana/Desktop/DATA/NH_dataset/cut_objects7/",
    'train': True,
    'val': True,
    'test': True,
    'data_suffix': "non_hw_only_50k_10_no_z_rot",
    'highway_files': False,
    'non_highway_files': False,
    'normalise': True,
    'random_seed': 42,
    'num_workers': 0,  # set number of cpu cores for data processing
    'plot_sample': True,
    'test_size': 0.1,
    'save_every': 10,
    'verbose': True,
    'normals': False,
    'eigenvalues': False,
    'random_id': random_id,
    'resume_from': 0,
    'resume_from_id': 0,
    'resume_model_path': f"C:/Users/Diana/PycharmProjects/pcdseg/runs/ObjectwiseSemSeg_92/epoch_mode_None_370_model.pth",
    # data processed_mode_1_traj_num_classes_6_prev
    'use_val_list': False,  # if false, random split with seed is used
    'val_list_path': "C:/Users/Diana/Desktop/DATA/Kitti360/kitti_for_dva/kitti360mm/raw/data_3d_semantics/2013_05_28_drive_val_reduced.txt"
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
        'rand_translate': 0.01,
        'rand_rotation_x': 0.15,
        'rand_rotation_y': 0.15,
        'rand_rotation_z': 0.15,
        'params_log_file': "params.json",
        'batch_norm': True,
        'loss_fn': 'nll'  # options: nll, focal
    },
    'SemSegmentation': {
        'lr': 0.005,
        'lr_decay': 0,
        'lr_cosine_step': 5,
        'mode': mode,  # 1, 2, 0
        'num_classes': separated_mode_class_nums[mode],
        'batch_size': 3,
        'num_epochs': 90,
        'subsample_to': 50000,
        'cut_in': 5,
        'rand_translate': 0.01,
        'rand_rotation_x': 0.15,
        'rand_rotation_y': 0.15,
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
        'label': 13,  # 0, 6, 11 - for NH
        'lr': 0.005,
        'lr_decay': 0,
        'lr_cosine_step': 10,  # if 0 - no scheduler applied
        'num_classes': 2,
        'batch_size': 4,
        'num_epochs': 1000,
        'subsample_to': 50000,
        'rand_translate': 0.01,
        'rand_rotation_x': 10,
        'rand_rotation_y': 10,
        'rand_rotation_z': 10,
        'params_log_file': "params.json",
        'batch_norm': True,
        'loss_fn': 'focal',  # options: nll, focal
    }
}
