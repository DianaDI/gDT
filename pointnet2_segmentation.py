import wandb
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from glob import glob
import random
import open3d as o3d
from collections import defaultdict
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.KITTI360Dataset import KITTI360Dataset
import json
import copy

from data.kitti_helpers import ground_label_ids
from data.utils import train_val_test_split, most_frequent
from model.pointnet2 import PointNet2


def remap_label_for_drawing(arr):
    arr = np.array(arr)
    unique_labels = np.unique(arr)
    label_dict = {unique_labels[j]: j for j in range(len(unique_labels))}
    arr = [label_dict[k] for k in arr]
    return arr


def cluster(xyz, eps=0.01, min_points=4):
    xyz = np.array(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    return cluster_labels


def get_normalised_weights():
    class_weights = (1 / torch.sqrt(torch.Tensor(list(train_dataset.class_weights_dict.values()))))
    max_w = max(class_weights)
    class_weights_normalised = class_weights / max_w
    return class_weights_normalised


def train(loader, save_dir, epoch, class_weights_normalised=None, save_every_epoch=5, log_every_iter=1):
    model.train()
    total_loss = correct_nodes = total_nodes = 0
    losses = []

    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = torch.squeeze(data.y).type(torch.LongTensor).to(device)
        loss = F.nll_loss(out, target)  # , weight=class_weights_normalised.to(device))
        # loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # scheduler.step()
        correct_nodes += out.argmax(dim=1).eq(target).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % log_every_iter == 0:
            train_acc = correct_nodes / total_nodes
            avg_loss = total_loss / log_every_iter
            print(f'[{i + 1}/{len(loader)}] Loss: {avg_loss:.4f} '
                  f'Train Acc: {train_acc:.4f}')
            wandb.log({'train_accuracy': train_acc,
                       "train_loss": loss.item()})
            losses.append(total_loss)
            total_loss = correct_nodes = total_nodes = 0
    if (epoch + 1) % save_every_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()},
            f'{save_dir}/epoch_mode_{config.mode}_{epoch + 1}_model.pth')
    return np.mean(losses)


@torch.no_grad()
def evaluate(loader, path=None, val_mode=True, log_img_every=5):
    iou_classwise, iou_classwise_new2 = {}, {}  # defaultdict(list)
    correct_nodes = total_nodes = 0
    accuracies = []
    if path:
        model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    ious_micro, ious_weighted, ious_macro, ious_weighted1, ious_weighted2 = list(), list(), list(), list(), list()
    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        out = out.cpu()
        target = torch.squeeze(data.y).cpu()
        labels_to_check = np.unique(target)
        if val_mode:
            val_loss = F.nll_loss(out, target.type(torch.LongTensor))
            wandb.log({'validation_loss': val_loss.item()})
        pred = np.argmax(out, axis=-1)
        iou_micro = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='micro')
        iou_macro = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='macro')
        iou_weighted = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='weighted')
        iou_classwise_temp = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average=None)
        for label, val in zip(labels_to_check, iou_classwise_temp):
            if label in iou_classwise.keys():
                iou_classwise[label].append(val)
            else:
                iou_classwise[label] = [val]
        ious_micro.append(iou_micro)
        ious_weighted.append(iou_weighted)
        ious_macro.append(iou_macro)
        # wandb.log({'miou_micro': iou_micro, 'miou_weighted': iou_weighted, 'miou_macro': iou_macro})

        correct_nodes += out.argmax(dim=1).eq(target).sum().item()
        total_nodes += data.num_nodes

        # if (i + 1) % 10 == 0:
        acc = correct_nodes / total_nodes
        print(f'[{i + 1}/{len(loader)}]'
              f'Eval Acc: {acc:.4f}')
        # wandb.log({'val_accuracy': acc}) if val_mode else wandb.log({'eval_accuracy': acc})
        accuracies.append(acc)
        correct_nodes = total_nodes = 0

        # do clustering
        xyz_filtered, cluster_labels = None, None

        if not val_mode:
            if config.eval_clustering:
                xyz = data.pos
                cond = (pred != 9) & (pred != 12) & (pred != 0) & (pred != 31)
                xyz_filtered = xyz[cond]  # filter out some classes
                pred_filtered = pred[cond]
                # target_filtered = target[cond]
                print('Clustering...')
                cluster_labels = cluster(xyz_filtered.cpu(), eps=0.03,
                                         min_points=5)  # not bad with 0,03 -> for pole +2%

                # iterate through clusters
                cluster_labels_unique = np.unique(cluster_labels)
                print(cluster_labels_unique)

                pred_fixed = copy.deepcopy(pred_filtered)
                for cl in cluster_labels_unique:
                    if cl == -1:
                        continue
                    # get most frequent label
                    # pred_under_cluster = pred_filtered[cluster_labels == cl]
                    # freq_label = most_frequent(pred_under_cluster)
                    freq_label = out[cond][cluster_labels == cl].sum(dim=0).argmax()
                    pred_fixed[cluster_labels == cl] = freq_label

                new_pred = copy.deepcopy(pred)
                new_pred[cond] = pred_fixed

                iou_classwise_temp2 = jaccard_score(y_true=target, y_pred=new_pred, labels=labels_to_check,
                                                    average=None)
                for label, val in zip(labels_to_check, iou_classwise_temp2):
                    if label in iou_classwise_new2.keys():
                        iou_classwise_new2[label].append(val)
                    else:
                        iou_classwise_new2[label] = [val]

                j1 = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check,
                                   average='weighted')
                ious_weighted1.append(j1)

                j2 = jaccard_score(y_true=target, y_pred=new_pred, labels=labels_to_check,
                                   average='weighted')
                ious_weighted2.append(j2)

            if config.verbose and (i + 1) % log_img_every == 0:
                pred_remapped = remap_label_for_drawing(pred)
                target_remapped = remap_label_for_drawing(data.y.cpu())
                wandb.log(
                    {
                        'eval_inputs': wandb.Object3D(
                            np.column_stack((np.array(data.pos.cpu()), np.array(data.x.cpu()) * 255))),
                        'eval_targets': wandb.Object3D(
                            np.column_stack((np.array(data.pos.cpu()), target_remapped))),
                        'eval_predictions': wandb.Object3D(
                            np.column_stack((np.array(data.pos.cpu()), pred_remapped))),
                        'clusters': wandb.Object3D(
                            np.column_stack((np.array(xyz_filtered.cpu()), cluster_labels + 1))),
                        'eval_accuracy': acc,
                        'miou_micro': iou_micro, 'miou_weighted': iou_weighted, 'miou_macro': iou_macro
                    })

    if not val_mode and config.eval_clustering:
        print("------")
        print('Initial:')
        print(np.mean(ious_weighted1))
        print('Filtered&Fixed:')
        print(np.mean(ious_weighted2))
        print("------")
        print("------")

        # print("Pred_fixed")
        # iou_classwise_new2 = dict(sorted(iou_classwise_new2.items()))
        # print(iou_classwise_new2)
        # for key, value in iou_classwise_new2.items():
        #     print(f'{key}: {np.mean(value)}')
        # print("------")

    return float(np.mean(ious_micro)), float(np.mean(ious_macro)), float(np.mean(ious_weighted)), \
           float(np.mean(accuracies)), \
           iou_classwise, iou_classwise_new2


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    with open("./data/class_label_counts.json", "r") as read_content:
        class_weights_dict_original = json.load(read_content)
    mode_class_nums = {0: 37,
                       1: len(ground_label_ids) + 1,
                       2: 33}  # s.non_ground_ids # len(all_label_ids) - len(ground_label_ids) + 1}

    wandb.finish()
    wandb.init(project="pointnet++")
    config = wandb.config
    config.learning_rate = 0.009
    config.epochs = 200
    config.cut_in = 4
    config.subsample_to = 50000
    config.mode = 2  # 0 - all, 1 - ground, 2 - non-ground
    config.scheduler_par = 1000
    config.seed = 402
    config.n_classes = mode_class_nums[config.mode]
    config.verbose = True
    config.resume_from = 0
    config.resume_from_rand_id = 0
    config.resume_model_path = None  # f'{ROOT_DIR}/runs/mode_{config.mode}_{config.resume_from_rand_id}/epoch_mode_{config.mode}_{config.resume_from}_model.pth'
    config.test = False
    config.train = True
    config.val = True
    config.eval_clustering = True
    config.batch_size = 1 if config.test else 3

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    train_path = "C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"

    all_files = sorted(glob(f"{train_path}*/static/*.ply"))
    train_files, test_files, val_files = train_val_test_split(all_files, seed=config.seed)  # train, test, val

    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.Compose([T.FixedPoints(config.subsample_to, replace=False),
                               T.NormalizeScale()])

    train_dataset = KITTI360Dataset(train_path, split="train", num_classes=config.n_classes, mode=config.mode,
                                    cut_in=config.cut_in,
                                    files=train_files, transform=transform, pre_transform=pre_transform)
    val_dataset = KITTI360Dataset(train_path, num_classes=config.n_classes, split="val", mode=config.mode,
                                  cut_in=config.cut_in,
                                  files=val_files, pre_transform=pre_transform)
    test_dataset = KITTI360Dataset(train_path, num_classes=config.n_classes, split="test", mode=config.mode,
                                   cut_in=config.cut_in,
                                   files=test_files, pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=7)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA AVAILABLE: {cuda_available}")
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if cuda_available else 'cpu')
    model = PointNet2(config.n_classes)
    model = model.to(device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Number of parameters: {params}")
    print(model)
    wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    save_dir = None
    if config.resume_from == 0 and not config.test:
        rand_num = random.randint(0, 1000)
        save_dir = f'{ROOT_DIR}/runs/mode_{config.mode}_{rand_num}'
        os.mkdir(save_dir)
        print(f'Save folder created: {os.path.isdir(save_dir)}')

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

        print(f'Saving in: {save_dir}')
        # scheduler = CosineAnnealingLR(optimizer, config.scheduler_par)
        loss_weights = get_normalised_weights()
        for epoch in tqdm(range(epoch_start, config.epochs)):
            loss = train(train_loader, save_dir, epoch, class_weights_normalised=loss_weights)
            if config.val:
                miou_micro, miou_macro, miou_weighted, mean_acc, classwise_iou, _ = evaluate(val_loader)
                print(f'Epoch: {epoch:02d}, Val IoU (w): {miou_weighted:.4f}')

    if config.test:
        print("RUNNING TEST...")
        model = model.to(device)
        miou_micro, miou_macro, miou_weighted, mean_acc, classwise_iou, iou_classwise_new2 = evaluate(test_loader,
                                                                                                      val_mode=False,
                                                                                                      log_img_every=1,
                                                                                                      path=f'{ROOT_DIR}/runs/mode_{config.mode}_964/epoch_mode_{config.mode}_600_model.pth')

        print(f'MODE {config.mode}')
        print(f'MIOU_MICRO: {miou_micro}')
        print(f'MIOU_MACRO: {miou_macro}')
        print(f'MIOU_WEIGHTED: {miou_weighted}')
        print(f'MEAN_ACCURACY: {mean_acc}')
        print("Old: ")
        classwise_iou = dict(sorted(classwise_iou.items()))
        print(classwise_iou)
        with open(glob(f"./mode{config.mode}_num_classes{config.n_classes}*.json")[0], "r") as read_content:
            id_name_dict = json.load(read_content)
        id_name_dict = dict([(int(k), id_name_dict[k]) for k in id_name_dict])
        print("original: ")
        for key in classwise_iou.keys():
            print(f'{id_name_dict[key]}: {np.mean(classwise_iou[key])}')
        print("Clustered: ")
        if iou_classwise_new2 is not None:
            for key in iou_classwise_new2.keys():
                print(f'{id_name_dict[key]}: {np.mean(iou_classwise_new2[key])}')
