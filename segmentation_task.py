import torch
import torch.nn.functional as F
import wandb
import numpy as np
from collections import defaultdict
from sklearn.metrics import jaccard_score
import open3d as o3d
import copy

from dl_task import DLTask


class SegmentationTask(DLTask):

    def train(self, loader, epoch, save_model_every_epoch=5):
        self.model.train()
        losses = []
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            target = torch.squeeze(data.y).type(torch.LongTensor).to(self.device)
            loss = F.nll_loss(out, target)  # , weight=class_weights_normalised.to(device))
            loss.backward()
            self.optimizer.step()
            # scheduler.step()
            accuracy = self.get_accuracy(out, target)
            wandb.log({"train_loss": loss.item(),
                       "train_acc": accuracy,
                       "train_iteration": step})
            print(f'[{i + 1}/{len(loader)}] Loss: {loss.item():.4f} '
                  f'Train Acc: {accuracy:.4f}')
            losses.append(loss.item())
        if (epoch + 1) % save_model_every_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                f'{self.model_save_dir}/epoch_mode_{self.config.mode}_{epoch + 1}_model.pth')
        return np.mean(losses)

    def compute_metrics(self, target, out, mode="val"):
        metrics_dict = defaultdict(list)
        iou_classwise = defaultdict(list)

        val_loss = F.nll_loss(out, target.type(torch.LongTensor))
        metrics_dict['loss'].append(val_loss)
        acc = self.get_accuracy(out, target)
        metrics_dict['accuracy'].append(acc)

        if mode != 'val':
            pred = np.argmax(out, axis=-1)
            labels_to_check = np.unique(target)
            iou_micro = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='micro')
            iou_macro = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='macro')
            iou_weighted = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average='weighted')
            metrics_dict['iou_micro'].append(iou_micro)
            metrics_dict['iou_macro'].append(iou_macro)
            metrics_dict['iou_weighted'].append(iou_weighted)

            iou_classwise_temp = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average=None)
            for label, val in zip(labels_to_check, iou_classwise_temp):
                iou_classwise[label].append(val)
        return metrics_dict, iou_classwise

    def dbscan_cluster(self, xyz, eps=0.01, min_points=4):
        xyz = np.array(xyz)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        return cluster_labels

    def compose_condition(self, a, vals):
        cond = (a != vals[0])
        for i in range(1, len(vals)):
            cond &= (a != vals[i])
        return cond

    def cluster(self, batch_data, out, target):
        xyz = batch_data.pos
        pred = np.argmax(out, axis=-1)
        cond = self.compose_condition(pred, [9, 12, 0,
                                             31])  # TODO move out this vals and double check if vals are correct after chaging ground classes
        xyz_filtered = xyz[cond]  # filter out above classes
        pred_filtered = pred[cond]
        cluster_labels = self.dbscan_cluster(xyz_filtered.cpu(), eps=0.03,
                                             min_points=5)  # not bad with 0,03 -> for pole +2%

        # iterate through clusters
        cluster_labels_unique = np.unique(cluster_labels)
        pred_fixed = copy.deepcopy(pred_filtered)
        for cl in cluster_labels_unique:
            if cl == -1:
                continue
            # get most frequent label
            freq_label = out[cond][cluster_labels == cl].sum(dim=0).argmax()
            pred_fixed[cluster_labels == cl] = freq_label

        new_pred = copy.deepcopy(pred)
        new_pred[cond] = pred_fixed
        metrics_dict, iou_classwise = self.compute_metrics(target, out, mode='eval')
        return metrics_dict, iou_classwise

    @torch.no_grad()
    def eval(self, loader, load_from_path=None, mode='val', epoch=0):
        if load_from_path:
            self.model.load_state_dict(torch.load(load_from_path)['model_state_dict'])
        self.model.eval()
        metrics_dict_all, metrics_dict_all_post = {}, {}
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            out = self.model(data)
            out = out.cpu()
            target = torch.squeeze(data.y).cpu()

            metrics_dict, iou_classwise = self.compute_metrics(target=target, out=out, mode=mode)
            metrics_dict_all = {key: metrics_dict.get(key, []) + metrics_dict_all.get(key, [])
                                for key in set(list(metrics_dict.keys()) + list(metrics_dict_all.keys()))}

            accuracy = metrics_dict["accuracy"][0]
            loss = metrics_dict['loss'][0]
            print(f'[{i + 1}/{len(loader)}]'
                  f'Eval Acc: {accuracy:.4f}')
            wandb.log({"val_loss": loss,
                       "val_acc": accuracy,
                       "val_iteration": step})

            if mode != 'val':
                if self.config.eval_clustering:
                    metrics_dict, iou_classwise = self.cluster(batch_data=data, out=out, target=target)
                    metrics_dict_all_post = {key: metrics_dict.get(key, []) + metrics_dict_all_post.get(key, [])
                                             for key in
                                             set(list(metrics_dict.keys()) + list(metrics_dict_all_post.keys()))}

                # if config.verbose and (i + 1) % log_img_every == 0:
                #     pred_remapped = remap_label_for_drawing(pred)
                #     target_remapped = remap_label_for_drawing(data.y.cpu())
                #     wandb.log(
                #         {
                #             'eval_inputs': wandb.Object3D(
                #                 np.column_stack((np.array(data.pos.cpu()), np.array(data.x.cpu()) * 255))),
                #             'eval_targets': wandb.Object3D(
                #                 np.column_stack((np.array(data.pos.cpu()), target_remapped))),
                #             'eval_predictions': wandb.Object3D(
                #                 np.column_stack((np.array(data.pos.cpu()), pred_remapped))),
                #             'clusters': wandb.Object3D(
                #                 np.column_stack((np.array(xyz_filtered.cpu()), cluster_labels + 1))),
                #             'eval_accuracy': acc,
                #             'miou_micro': iou_micro, 'miou_weighted': iou_weighted, 'miou_macro': iou_macro
                #         })

        return metrics_dict_all, metrics_dict_all_post
