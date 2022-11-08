import os
import random

from data.kitti_helpers import ground_label_ids

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"

random_id = random.randint(0, 1000)

mode = 0

COMMON_PARAMS = {
    'train': True,
    'val': True,
    'test': False,
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
    'resume_model_path': f"C:/Users/Diana/PycharmProjects/pcdseg/runs/SemSegmentation_873/epoch_mode_{mode}_90_model.pth"
        # "C:/Users/Diana/PycharmProjects/pcdseg/runs/SemSegmentation_620/epoch_mode_2_200_model.pth"
    # "C:/Users/Diana/PycharmProjects/pcdseg/runs/SemSegmentation_234/epoch_mode_2_200_model.pth"
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
        'rand_rotation_x': 0,
        'rand_rotation_y': 0,
        'rand_rotation_z': 0,
        'params_log_file': "params.json",
        'batch_norm': True,
        'loss_fn': 'nll'  # options: nll, focal
    },
    'SemSegmentation': {
        'lr': 0.001,
        'lr_decay': 0,
        'lr_cosine_step': 10,
        'mode': mode,  # 1, 2, 0
        'num_classes': separated_mode_class_nums[mode],
        'batch_size': 1 if COMMON_PARAMS['test'] else 3,
        'num_epochs': 100,
        'subsample_to': 50000,
        'cut_in': 5,
        'rand_translate': 0.01,
        'rand_rotation_x': 0,
        'rand_rotation_y': 0,
        'rand_rotation_z': 0,
        'params_log_file': "params.json",
        'eval_clustering': False,
        'batch_norm': True,
        'loss_fn': 'focal',  # options: nll, focal
        'clustering_eps': 0.005,  # 0.014, for mode 2
        'clustering_min_points': 30  # 4 for mode 2
    }
}

# todo: run 3 evals
# mode 2: 620
# mode 0: 854
# mode 1: 873
