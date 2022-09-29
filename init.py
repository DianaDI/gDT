import os

from data.kitti_helpers import ground_label_ids

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"

# train_files = open(f"{TRAIN_PATH}2013_05_28_drive_train.txt", "r").read().splitlines()
# val_files = open(f"{TRAIN_PATH}2013_05_28_drive_val.txt", "r").read().splitlines()

# root = "C:/Users/Diana/Desktop/DATA/Kitti360/"
# train_files = [root + i for i in train_files]
# val_files = [root + i for i in val_files]


COMMON_PARAMS = {
    'train': True,
    'val': True,
    'test': False,
    'normalise': True,
    'random_seed': 402,
    'num_workers': 7,  # set number of cpu cores for data processing
    'plot_sample': True,
    'test_size': 0.1,
    'save_every': 10,
    'verbose': True,
    'resume_from': 0,
    'resume_from_id': 0,
    'resume_model_path': None  # "C:/Users/Diana/PycharmProjects/pcdseg/runs/binary_294/epoch_280_model.pth"
}

separated_mode_class_nums = {0: 37,
                             1: len(ground_label_ids) + 1,
                             2: 33}

MODEL_SPECIFIC_PARAMS = {
    'GroundDetection': {
        'lr': 0.001,
        'lr_decay': 0.99,  # every epoch
        'lr_cosine_step': None,
        'batch_size': 4,
        'num_epochs': 200,
        'subsample_to': 50000,
        'cut_in': 4,
        'num_classes': 2,
        'rand_translate': 0.01,
        'rand_rotation_x': 0,
        'rand_rotation_y': 0,
        'rand_rotation_z': 0,
        'params_log_file': "params.json"
    },
    'SemSegmentation': {
        'lr': 0.003,
        'lr_decay': 0.99,
        'lr_cosine_step': 1000,
        'mode': 2,  # 1, 2
        'num_classes': 33,
        'batch_size': 3,
        'num_epochs': 200,
        'subsample_to': 50000,
        'cut_in': 4,
        'rand_translate': 0.01,
        'rand_rotation_x': 15,
        'rand_rotation_y': 15,
        'rand_rotation_z': 15,
        'params_log_file': "params.json",
        'eval_clustering': True
    }
}
