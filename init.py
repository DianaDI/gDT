import os
import random

from data.kitti_helpers import ground_label_ids

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
linux = False
TRAIN_PATH = "/rds/user/dd593/hpc-work/data_3d_semantics/train/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"
GROUND_SEP_ROOT = "/rds/user/dd593/hpc-work/inliers_traj_0.6/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train_processed/inliers_traj_0.6"
POSES_DIR = "/rds/user/dd593/hpc-work/data_poses/" if linux else "C:/Users/Diana/Desktop/DATA/Kitti360/data_poses"

random_id = random.randint(0, 1000)

mode = 0

COMMON_PARAMS = {
    'train': True,
    'val': True,
    'test': True,
    'highway_files': True,
    'normalise': True,
    'random_seed': 402,
    'num_workers': 7,  # set number of cpu cores for data processing
    'plot_sample': True,
    'test_size': 0.1,
    'save_every': 5,
    'verbose': True,
    'normals': True,
    'eigenvalues': True,
    'random_id': random_id,
    'resume_from': 0,
    'resume_from_id': 0,
    'resume_model_path': f"C:/Users/Diana/PycharmProjects/pcdseg/runs/SemSegmentation_121/epoch_mode_{mode}_90_model.pth"
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
        'num_epochs': 100,
        'subsample_to': 70000,
        'cut_in': 2,
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
        'lr': 0.001,
        'lr_decay': 0,
        'lr_cosine_step': 5,
        'mode': mode,  # 1, 2, 0
        'num_classes': separated_mode_class_nums[mode],
        'batch_size': 3,
        'num_epochs': 150,
        'subsample_to': 50000,
        'cut_in': 5,
        'rand_translate': 0.01,
        'rand_rotation_x': 0.30,
        'rand_rotation_y': 0.30,
        'rand_rotation_z': 0.05,
        'params_log_file': "params.json",
        'eval_clustering': False,
        'batch_norm': True,
        'loss_fn': 'focal',  # options: nll, focal
        'clustering_eps': 0.025,  # 0.014, for mode 2
        'clustering_min_points': 30,  # 4 for mode 2
        'ignore_labels': True  # now works only with focal loss
    }
}
