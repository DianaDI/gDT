import wandb
import numpy as np
import os
import torch
from tqdm import tqdm
from glob import glob
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import json
from torchgeometry.losses import FocalLoss

from model.pointnet2 import PointNet2
from init import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS, TRAIN_PATH, ROOT_DIR, GROUND_SEP_ROOT, POSES_DIR
from data.KITTI360DatasetBinary import KITTI360DatasetBinary
from data.KITTI360Dataset import KITTI360Dataset, HIGHWAY_SCENES_FILES
from data.utils import train_val_test_split, get_ignore_labels, get_train_val_test_split_from_file
from segmentation_task import SegmentationTask
from data.transforms import NormalizeFeatureToMeanStd

if __name__ == '__main__':
    task_name = 'SemSegmentation'  # 'GroundDetection'
    params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[task_name]}

    DatasetClass = KITTI360Dataset if task_name == 'SemSegmentation' else KITTI360DatasetBinary

    wandb.finish()
    wandb.init(project=task_name)
    # wandb.init()

    wandb.define_metric("train_iteration")
    wandb.define_metric("val_iteration")

    wandb.define_metric("train_loss", step_metric="train_iteration")
    wandb.define_metric("val_loss", step_metric="val_iteration")
    wandb.define_metric("train_acc", step_metric="train_iteration")
    wandb.define_metric("val_acc", step_metric="val_iteration")

    cfg = wandb.config
    cfg.learning_rate = params['lr']
    cfg.mode = params['mode']
    cfg.epochs = params['num_epochs']
    cfg.cut_in = params['cut_in']
    cfg.subsample_to = params['subsample_to']
    cfg.seed = params['random_seed']
    cfg.n_classes = params['num_classes']
    cfg.verbose = params['verbose']
    cfg.resume_from = params['resume_from']
    cfg.resume_from_id = params['resume_from_id']
    cfg.resume_model_path = params['resume_model_path']
    cfg.test = params['test']
    cfg.train = params['val']
    cfg.val = params['train']
    cfg.batch_size = params['batch_size']
    cfg.lr_decay = params['lr_decay']
    cfg.lr_cosine_step = params['lr_cosine_step']
    cfg.params_log_file = params['params_log_file']
    cfg.batch_norm = params['batch_norm']
    cfg.loss_fn = params['loss_fn']
    cfg.random_id = params['random_id']
    cfg.normals = params['normals']
    cfg.eigenvalues = params['eigenvalues']
    cfg.ignore_labels = params['ignore_labels']
    cfg.save_every = params['save_every']
    cfg.highway_files = params['highway_files']
    cfg.non_highway_files = params['non_highway_files']
    cfg.data_suffix = params['data_suffix']
    cfg.use_val_list = params['use_val_list']
    cfg.val_list_path = params['val_list_path']
    cfg.load_predictions = params['load_predictions']

    if task_name == 'SemSegmentation':
        cfg.eval_clustering = params['eval_clustering']
        cfg.clustering_eps = params['clustering_eps']
        cfg.clustering_min_points = params['clustering_min_points']

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    path = TRAIN_PATH
    all_files = []
    if cfg.highway_files:
        print("TRAINING WITH HIGHWAY FILES ONLY")
        for i in HIGHWAY_SCENES_FILES:
            all_files.append(os.path.join(path, i))
    elif cfg.non_highway_files:
        temp_all_files = sorted(glob(f"{path}*/static/*.ply"))
        h_files = []
        for i in HIGHWAY_SCENES_FILES:
            h_files.append(os.path.join(path, i))
        for f in temp_all_files:
            if f not in h_files:
                all_files.append(f)
    else:
        all_files = sorted(glob(f"{path}*/static/*.ply"))

    if cfg.use_val_list:
        train_files, test_files, val_files = get_train_val_test_split_from_file(split_path=cfg.val_list_path,
                                                                                data_root_path=path,
                                                                                all_files=all_files, seed=cfg.seed)
    else:
        train_files, test_files, val_files = train_val_test_split(all_files, seed=cfg.seed)

    transforms = []
    if params['rand_translate'] > 0:
        transforms.append(T.RandomJitter(params['rand_translate']))
    if params['rand_rotation_x'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_x'], axis=0))
    if params['rand_rotation_y'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_y'], axis=1))
    if params['rand_rotation_z'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_z'], axis=2))

    transform = T.Compose(transforms)
    pre_transform = T.Compose([  # T.FixedPoints(cfg.subsample_to, replace=False),
        T.NormalizeScale(),
        # NormalizeFeatureToMeanStd()
    ])

    print(f'MODE: {cfg.mode}')

    train_dataset = DatasetClass(path, split="train", num_classes=cfg.n_classes, mode=cfg.mode,
                                 cut_in=cfg.cut_in, normals=cfg.normals, eigenvalues=cfg.eigenvalues,
                                 files=train_files, transform=transform, pre_transform=pre_transform,
                                 ground_points_dir=GROUND_SEP_ROOT, poses_dir=POSES_DIR, config=cfg)
    val_dataset = DatasetClass(path, num_classes=cfg.n_classes, split="val", mode=cfg.mode,
                               cut_in=cfg.cut_in, normals=cfg.normals, eigenvalues=cfg.eigenvalues,
                               files=val_files, pre_transform=pre_transform, ground_points_dir=GROUND_SEP_ROOT,
                               poses_dir=POSES_DIR, config=cfg)
    test_dataset = DatasetClass(path, num_classes=cfg.n_classes, split="test", mode=cfg.mode,
                                cut_in=cfg.cut_in, normals=cfg.normals, eigenvalues=cfg.eigenvalues,
                                files=test_files, pre_transform=pre_transform, ground_points_dir=GROUND_SEP_ROOT,
                                poses_dir=POSES_DIR, config=cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=params['num_workers'])

    cuda_available = torch.cuda.is_available()
    print(f"CUDA AVAILABLE: {cuda_available}")
    print(torch.cuda.get_device_name(0))

    feature_channels = 3  # default RGB
    if cfg.normals:
        feature_channels += 3
    if cfg.eigenvalues:
        feature_channels += 3

    device = torch.device('cuda' if cuda_available else 'cpu')
    model = PointNet2(cfg.n_classes, batch_norm=cfg.batch_norm, feature_channels=feature_channels)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {model_params}")
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer,
                                  cfg.lr_cosine_step, verbose=True) if cfg.lr_cosine_step > 0 else None
    # ExponentialLR(optimizer, config.lr_decay, verbose=config.verbose)

    loss_fn = FocalLoss(alpha=0.5, gamma=2.0,
                        reduction='mean') if cfg.loss_fn == 'focal' else None  # default nll defined inside methods
    save_dir = None
    if cfg.resume_from == 0 and cfg.train:
        rand_num = cfg.random_id
        save_dir = f'{ROOT_DIR}/runs/{task_name}_{rand_num}'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Save folder created: {os.path.isdir(save_dir)}')
        else:
            print("Repeated random id, run again")
        with open(f'{save_dir}/{cfg.params_log_file}', 'w') as fp:
            json.dump(params, fp, indent=2)

    dl_task = SegmentationTask(name=task_name, device=device, model=model, scheduler=scheduler, mode=cfg.mode,
                               num_classes=cfg.n_classes,
                               model_save_dir=save_dir, optimizer=optimizer, config=cfg)
    wandb.watch(dl_task.model)

    ignored_labels = get_ignore_labels(mode=cfg.mode) if cfg.ignore_labels else None

    if cfg.train:
        print("RUNNING TRAINING...")
        epoch_start = 0
        if cfg.resume_from > 0:
            print("RESUMING PREV TRAINING...")
            checkpoint = torch.load(cfg.resume_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            dl_task.model = model
            dl_task.optimizer = optimizer
            epoch_start = checkpoint['epoch'] + 1
            save_dir = os.path.dirname(cfg.resume_model_path)
            dl_task.model_save_dir = save_dir

        print(f'Saving in: {save_dir}')
        for epoch in tqdm(range(epoch_start, cfg.epochs)):
            train_loss = dl_task.train(loader=train_loader, loss_fn=loss_fn, epoch=epoch,
                                       save_model_every_epoch=cfg.save_every, ignored_labels=ignored_labels)
            if cfg.val:
                metrics_dict, _ = dl_task.eval(loader=val_loader, loss_fn=loss_fn, epoch=epoch)
                print(f'Epoch: {epoch:02d}, Mean acc: {np.mean(metrics_dict["accuracy"]):.4f}')

    if cfg.test:
        print("RUNNING TEST...")
        load_from_path = None if cfg.train else cfg.resume_model_path
        metrics_dict, extra_metrics = dl_task.eval(loader=test_loader, loss_fn=loss_fn, load_from_path=load_from_path,
                                                   mode='eval')
        print(f'MEAN_ACCURACY: {np.mean(metrics_dict["accuracy"])}')
        dl_task.print_res(metrics_dict, 'ALL METRICS (no clustering)', print_overall_mean=False,
                          mean_over_nonzero=False)
