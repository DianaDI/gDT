import os

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
    'resume_model_path': None # "C:/Users/Diana/PycharmProjects/pcdseg/runs/binary_294/epoch_280_model.pth"
}

MODEL_SPECIFIC_PARAMS = {
    'GroundDetection': {
        'lr': 0.01,
        'lr_decay': 0.99,  # every epoch
        'batch_size': 1,
        'num_epochs': 200,
        'subsample_to': 200000,
        'cut_in': 2,
        'num_classes': 2,
        'rand_translate': 0.01,
        'rand_rotation_x': 0,
        'rand_rotation_y': 0,
        'rand_rotation_z': 0,
        'params_log_file': "params.json"
    },
    'SemSegmentation': {
        'lr': 0.0001,
        'mode': 0  # 1, 2
    }
}
