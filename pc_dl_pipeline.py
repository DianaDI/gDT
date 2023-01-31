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
from data.utils import train_val_test_split, get_ignore_labels
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

    config = wandb.config
    config.learning_rate = params['lr']
    config.mode = params['mode']
    config.epochs = params['num_epochs']
    config.cut_in = params['cut_in']
    config.subsample_to = params['subsample_to']
    config.seed = params['random_seed']
    config.n_classes = params['num_classes']
    config.verbose = params['verbose']
    config.resume_from = params['resume_from']
    config.resume_from_id = params['resume_from_id']
    config.resume_model_path = params['resume_model_path']
    config.test = params['test']
    config.train = params['val']
    config.val = params['train']
    config.batch_size = params['batch_size']
    config.lr_decay = params['lr_decay']
    config.lr_cosine_step = params['lr_cosine_step']
    config.params_log_file = params['params_log_file']
    config.batch_norm = params['batch_norm']
    config.loss_fn = params['loss_fn']
    config.random_id = params['random_id']
    config.normals = params['normals']
    config.eigenvalues = params['eigenvalues']
    config.ignore_labels = params['ignore_labels']
    config.save_every = params['save_every']
    config.highway_files = params['highway_files']
    config.data_suffix = params['data_suffix']

    if task_name == 'SemSegmentation':
        config.eval_clustering = params['eval_clustering']
        config.clustering_eps = params['clustering_eps']
        config.clustering_min_points = params['clustering_min_points']

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    path = TRAIN_PATH
    all_files = []
    if config.highway_files:
        print("TRAINING WITH HIGHWAY FILES ONLY")
        for i in HIGHWAY_SCENES_FILES:
            all_files.append(os.path.join(path, i))
    else:
        all_files = sorted(glob(f"{path}*/static/*.ply"))
    train_files, test_files, val_files = train_val_test_split(all_files, seed=config.seed)

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
    pre_transform = T.Compose([T.FixedPoints(config.subsample_to, replace=False),
                               T.NormalizeScale(),
                               NormalizeFeatureToMeanStd()])

    print(f'MODE: {config.mode}')

    train_dataset = DatasetClass(path, split="train", num_classes=config.n_classes, mode=config.mode,
                                 cut_in=config.cut_in, normals=config.normals, eigenvalues=config.eigenvalues,
                                 files=train_files, transform=transform, pre_transform=pre_transform,
                                 ground_points_dir=GROUND_SEP_ROOT, poses_dir=POSES_DIR, config=config)
    val_dataset = DatasetClass(path, num_classes=config.n_classes, split="val", mode=config.mode,
                               cut_in=config.cut_in, normals=config.normals, eigenvalues=config.eigenvalues,
                               files=val_files, pre_transform=pre_transform, ground_points_dir=GROUND_SEP_ROOT,
                               poses_dir=POSES_DIR, config=config)
    test_dataset = DatasetClass(path, num_classes=config.n_classes, split="test", mode=config.mode,
                                cut_in=config.cut_in, normals=config.normals, eigenvalues=config.eigenvalues,
                                files=test_files, pre_transform=pre_transform, ground_points_dir=GROUND_SEP_ROOT,
                                poses_dir=POSES_DIR, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=params['num_workers'])

    cuda_available = torch.cuda.is_available()
    print(f"CUDA AVAILABLE: {cuda_available}")
    print(torch.cuda.get_device_name(0))

    feature_channels = 3  # default RGB
    if config.normals:
        feature_channels += 3
    if config.eigenvalues:
        feature_channels += 3

    device = torch.device('cuda' if cuda_available else 'cpu')
    model = PointNet2(config.n_classes, batch_norm=config.batch_norm, feature_channels=feature_channels)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {model_params}")
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer,
                                  config.lr_cosine_step, verbose=True) if config.lr_cosine_step > 0 else None
    # ExponentialLR(optimizer, config.lr_decay, verbose=config.verbose)

    loss_fn = FocalLoss(alpha=0.5, gamma=2.0,
                        reduction='mean') if config.loss_fn == 'focal' else None  # default nll defined inside methods
    save_dir = None
    if config.resume_from == 0 and config.train:
        rand_num = config.random_id
        save_dir = f'{ROOT_DIR}/runs/{task_name}_{rand_num}'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Save folder created: {os.path.isdir(save_dir)}')
        else:
            print("Repeated random id, run again")
        with open(f'{save_dir}/{config.params_log_file}', 'w') as fp:
            json.dump(params, fp, indent=2)

    dl_task = SegmentationTask(name=task_name, device=device, model=model, scheduler=scheduler, mode=config.mode,
                               num_classes=config.n_classes,
                               model_save_dir=save_dir, optimizer=optimizer, config=config)
    wandb.watch(dl_task.model)

    ignored_labels = get_ignore_labels(mode=config.mode) if config.ignore_labels else None

    if config.train:
        print("RUNNING TRAINING...")
        epoch_start = 0
        if config.resume_from > 0:
            print("RESUMING PREV TRAINING...")
            checkpoint = torch.load(config.resume_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            dl_task.model = model
            dl_task.optimizer = optimizer
            epoch_start = checkpoint['epoch'] + 1
            save_dir = os.path.dirname(config.resume_model_path)
            dl_task.model_save_dir = save_dir

        print(f'Saving in: {save_dir}')
        for epoch in tqdm(range(epoch_start, config.epochs)):
            train_loss = dl_task.train(loader=train_loader, loss_fn=loss_fn, epoch=epoch,
                                       save_model_every_epoch=config.save_every, ignored_labels=ignored_labels)
            if config.val:
                metrics_dict, _ = dl_task.eval(loader=val_loader, loss_fn=loss_fn, epoch=epoch)
                print(f'Epoch: {epoch:02d}, Mean acc: {np.mean(metrics_dict["accuracy"]):.4f}')

    if config.test:
        print("RUNNING TEST...")
        load_from_path = None if config.train else config.resume_model_path
        metrics_dict, extra_metrics = dl_task.eval(loader=test_loader, loss_fn=loss_fn, load_from_path=load_from_path,
                                                   mode='eval')
        print(f'MEAN_ACCURACY: {np.mean(metrics_dict["accuracy"])}')
        dl_task.print_res(metrics_dict, 'ALL METRICS (no clustering)', print_overall_mean=False,
                          mean_over_nonzero=False)
