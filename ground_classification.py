import wandb
import numpy as np
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from glob import glob
import random
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import json
from sklearn.metrics import precision_recall_fscore_support

from model.pointnet2 import PointNet2
from init import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS, TRAIN_PATH, ROOT_DIR
from data.KITTI360DatasetBinary import KITTI360DatasetBinary
from data.utils import train_val_test_split


def get_accuracy(out, target):
    correct_nodes = out.argmax(dim=1).eq(target).sum().item()
    return correct_nodes / len(target)


def train(loader, save_dir, epoch, save_every_epoch=10):
    model.train()
    losses = []

    for i, data in enumerate(loader):
        step = i + len(loader) * epoch
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = torch.squeeze(data.y).type(torch.LongTensor).to(device)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # target_remapped = remap_label_for_drawing(data[0].y.cpu())
        train_acc = get_accuracy(out, target)
        wandb.log({"train_loss": loss.item(),
                   "train_acc": train_acc,
                   "train_iteration": step})
        print(f'[{i + 1}/{len(loader)}] Loss: {loss.item():.4f} '
              f'Train Acc: {train_acc:.4f}')

    if (epoch + 1) % save_every_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f'{save_dir}/epoch_{epoch + 1}_model.pth')

    scheduler.step()
    return np.mean(losses)


@torch.no_grad()
def evaluate(loader, epoch=0, path=None, model=None, val_mode=True, log_img_every=5):
    accuracies, precs0, precs1, recalls0, recalls1 = [], [], [], [], []
    if path:
        model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    losses = []
    for i, data in enumerate(loader):
        step = i + len(loader) * epoch
        data = data.to(device)
        out = model(data)
        out = out.cpu()
        target = torch.squeeze(data.y).cpu()
        val_loss = F.nll_loss(out, target.type(torch.LongTensor))
        losses.append(val_loss.item())
        acc = get_accuracy(out, target)
        precision, recall, _, _ = precision_recall_fscore_support(y_true=target, y_pred=out.argmax(dim=1))
        print(f'Precision: {precision}, Recall {recall}')
        wandb.log({"val_loss": val_loss.item(),
                   "val_acc": acc,
                   "val_iteration": step})
        print(f'[{i + 1}/{len(loader)}]'
              f'Eval Acc: {acc:.4f}')
        accuracies.append(acc)
        precs0.append(precision[0])
        precs1.append(precision[1])
        recalls0.append(recall[0])
        recalls1.append(recall[1])

    return float(np.mean(accuracies)), np.mean(losses), np.mean(precs0), np.mean(precs1), np.mean(recalls0), np.mean(
        recalls1)


if __name__ == '__main__':
    wandb.finish()
    wandb.init(project="pointnet++")
    wandb.init()

    wandb.define_metric("train_iteration")
    wandb.define_metric("val_iteration")

    wandb.define_metric("train_loss", step_metric="train_iteration")
    wandb.define_metric("val_loss", step_metric="val_iteration")
    wandb.define_metric("train_acc", step_metric="train_iteration")
    wandb.define_metric("val_acc", step_metric="val_iteration")

    model_class = 'GroundDetection'
    params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[model_class]}

    config = wandb.config
    config.learning_rate = params['lr']
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
    config.params_log_file = params['params_log_file']

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    path = TRAIN_PATH
    all_files = sorted(glob(f"{path}*/static/*.ply"))
    train_files, test_files, val_files = train_val_test_split(all_files, seed=config.seed)

    transforms = []
    if params['rand_translate'] > 0:
        transforms.append(T.RandomTranslate(params['rand_translate']))
    if params['rand_rotation_x'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_x'], axis=0))
    if params['rand_rotation_y'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_y'], axis=1))
    if params['rand_rotation_z'] > 0:
        transforms.append(T.RandomRotate(params['rand_rotation_z'], axis=2))

    transform = T.Compose(transforms)
    pre_transform = T.Compose([T.FixedPoints(config.subsample_to, replace=False),
                               T.NormalizeScale()])

    train_dataset = KITTI360DatasetBinary(path, split="train",
                                          cut_in=config.cut_in,
                                          files=train_files, transform=transform, pre_transform=pre_transform)
    val_dataset = KITTI360DatasetBinary(path, split="val",
                                        cut_in=config.cut_in,
                                        files=val_files, pre_transform=pre_transform)
    test_dataset = KITTI360DatasetBinary(path, split="test",
                                         cut_in=config.cut_in,
                                         files=test_files, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=params['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=params['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=params['num_workers'])

    cuda_available = torch.cuda.is_available()
    print(f"CUDA AVAILABLE: {cuda_available}")
    print(torch.cuda.get_device_name(0))

    device = torch.device('cuda' if cuda_available else 'cpu')
    model = PointNet2(config.n_classes)
    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {model_params}")
    # print(model)
    wandb.watch(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    save_dir = None
    if config.resume_from == 0 and not config.test:
        rand_num = random.randint(0, 1000)
        save_dir = f'{ROOT_DIR}/runs/binary_{rand_num}'
        os.mkdir(save_dir)
        print(f'Save folder created: {os.path.isdir(save_dir)}')
        with open(f'{save_dir}/{config.params_log_file}', 'w') as fp:
            json.dump(params, fp, indent=2)

    if config.train:
        print("RUNNING TRAINING...")
        epoch_start = 0
        if config.resume_from > 0:
            print("RESUMING PREV TRAINING...")
            checkpoint = torch.load(config.resume_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch'] + 1
            save_dir = os.path.dirname(config.resume_model_path)

            # with open(f'{save_dir}/{config.params_log_file}', 'w') as fp:
            #     json.dump(params, fp)

        print(f'Saving in: {save_dir}')
        scheduler = ExponentialLR(optimizer, config.lr_decay, verbose=config.verbose)
        for epoch in tqdm(range(epoch_start, config.epochs)):
            train_loss = train(train_loader, save_dir, epoch)
            if config.val:
                mean_acc, val_loss, _, _, _, _ = evaluate(val_loader, model=model, epoch=epoch)
                print(f'Epoch: {epoch:02d}, Acc: {mean_acc:.4f}')

    if config.test:
        print("RUNNING TEST...")
        model = model.to(device)
        mean_acc = evaluate(test_loader,
                            model=model,
                            val_mode=False,
                            log_img_every=1,
                            path=config.resume_model_path)

        print(f'MEAN_ACCURACY: {mean_acc}')
