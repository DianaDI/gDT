import torch
import torch.nn.functional as F
import wandb
import numpy as np
from collections import defaultdict
from sklearn.metrics import jaccard_score
import open3d as o3d
import copy
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torchgeometry.losses import FocalLoss

from dl_task import DLTask
from metrics.confusion_matrix import ConfusionMatrix
from data.pcd_utils import torchdata2o3dpcd, cluster_with_intensities, draw_pc_with_labels


class SegmentationTask(DLTask):

    def train(self, loader, epoch, loss_fn=None, save_model_every_epoch=5, ignored_labels=None):
        self.model.train()
        losses = []
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            target = torch.squeeze(data.y).type(torch.LongTensor).to(self.device)
            if ignored_labels is None:
                loss = F.nll_loss(out, target) if not loss_fn else loss_fn(out.t().unsqueeze(0).unsqueeze(2),
                                                                           target.unsqueeze(0).unsqueeze(1))
            else:
                out_t = out.t().unsqueeze(0).unsqueeze(2)
                target_t = target.unsqueeze(0).unsqueeze(1)
                losses_no_reduct = FocalLoss(alpha=0.5, gamma=2.0, reduction='none')(out_t, target_t)
                weights = torch.logical_not(sum(target_t.squeeze() == i for i in ignored_labels).bool()).int()
                losses_w = losses_no_reduct * weights
                loss = torch.mean(losses_w)

            loss.backward()
            self.optimizer.step()
            accuracy = self.get_accuracy(out, target)
            wandb.log({"train_loss": loss.item(),
                       "train_acc": accuracy,
                       "train_iteration": step
                       # 'eval_inputs': wandb.Object3D(
                       #     np.column_stack((np.array(data[0].pos.cpu()), np.array(data[0].x[:, :3].cpu()) * 255)))
                       })
            if (i + 1) % 10 == 0:
                print(f'[{i + 1}/{len(loader)}] Loss: {loss.item():.4f} '
                      f'Train Acc: {accuracy:.4f}')
            losses.append(loss.item())
        if (epoch + 1) % save_model_every_epoch == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                f'{self.model_save_dir}/epoch_mode_{self.config.mode}_{epoch + 1}_model.pth')
        if self.scheduler:
            self.scheduler.step()
        return np.mean(losses)

    def compute_metrics(self, target, out, pred, loss_fn=None, mode="val"):
        metrics_dict = defaultdict(list)
        iou_classwise = defaultdict(list)

        loss = F.nll_loss(out, target) if not loss_fn else loss_fn(out.t().unsqueeze(0).unsqueeze(2),
                                                                   target.unsqueeze(0).unsqueeze(1))
        metrics_dict['loss'].append(loss)
        acc = self.get_accuracy(out, target)
        metrics_dict['accuracy'].append(acc)

        if mode != 'val':
            labels_to_check = np.unique(target)
            iou_micro = jaccard_score(y_true=target, y_pred=pred, average='micro')
            iou_macro = jaccard_score(y_true=target, y_pred=pred, average='macro')
            iou_weighted = jaccard_score(y_true=target, y_pred=pred, average='weighted')
            metrics_dict['iou_micro'].append(iou_micro)
            metrics_dict['iou_macro'].append(iou_macro)
            metrics_dict['iou_weighted'].append(iou_weighted)

            iou_classwise_temp = jaccard_score(y_true=target, y_pred=pred, labels=labels_to_check, average=None)
            for label, val in zip(labels_to_check, iou_classwise_temp):
                iou_classwise[label].append(val)

        return metrics_dict, iou_classwise

    def dbscan_cluster_o3d(self, xyz, rgb, eps=0.01, min_points=4):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
        pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        return cluster_labels

    def dbscan_cluster_sklearn(self, xyz, rgb, eps=0.014, min_points=20):
        # X = np.column_stack((xyz, rgb))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        print(f'DIST AVG: {avg_dist}')
        X = xyz
        eps = avg_dist * 9  # 10, 9 is nice with min points=20
        db = DBSCAN(eps=eps, min_samples=17).fit(X)
        return db.labels_

    def compose_condition(self, a, vals):
        cond = (a != vals[0])
        for i in range(1, len(vals)):
            cond &= (a != vals[i])
        return cond

    def draw_pc_with_labels(self, xyz, labels):
        max_label = labels.max()
        # print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("hsv")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.points = o3d.utility.Vector3dVector(xyz)

        o3d.visualization.draw_geometries([pcd])

    def remap_label_for_drawing(self, arr):
        arr = np.array(arr)
        unique_labels = np.unique(arr)
        label_dict = {unique_labels[j]: j for j in range(len(unique_labels))}
        arr = [label_dict[k] for k in arr]
        return arr

    def compute_under_clusters(self, cluster_labels, cluster_labels_unique, pred):
        print(f'Clusters cnt: {cluster_labels_unique}')
        new_pred = copy.deepcopy(pred)
        for cl in cluster_labels_unique:
            if cl == 0:
                continue
            # get most frequent label
            pred_under_cluster = list(pred[cluster_labels == cl])
            freq_label = Counter(pred_under_cluster).most_common(1)[0][
                0]  # max(set(pred_under_cluster), key=pred_under_cluster.count)
            new_pred[cluster_labels == cl] = freq_label
        return new_pred

    def cluster_and_apply_frequent_label(self, batch_data, out, target):
        xyz = batch_data.pos
        rgb = batch_data.x
        pred = np.argmax(out, axis=-1)
        cond = self.compose_condition(pred, [
            10])  # TODO move out this vals and double check if vals are correct after chaging ground classes
        xyz_filtered = xyz.cpu()  # [cond].cpu()  # filter out above classes
        pred_filtered = pred.cpu()  # [cond].cpu()
        rgb_filtered = rgb.cpu()  # [cond].cpu()
        # cluster_labels = self.dbscan_cluster_o3d(xyz_filtered.cpu(), rgb_filtered.cpu(), eps=0.025,
        #                                          min_points=50)  # not bad with 0,03 -> for pole +2%

        metrics_dict, iou_classwise = None, None
        if len(xyz_filtered) > 0:
            cluster_labels = self.dbscan_cluster_sklearn(xyz_filtered, rgb_filtered, eps=self.config.clustering_eps,
                                                         min_points=self.config.clustering_min_points)

            # self.draw_pc_with_labels(xyz_filtered, cluster_labels)
            # self.draw_pc_with_labels(xyz_filtered, pred_filtered)

            # wandb.log(
            #     {'clusters': wandb.Object3D(
            #             np.column_stack((np.array(xyz_filtered.cpu()), cluster_labels + 1)))})

            # iterate through clusters
            cluster_labels_unique = np.unique(cluster_labels)
            print(f'Clusters cnt: {cluster_labels_unique}')
            pred_fixed = copy.deepcopy(pred_filtered)
            for cl in cluster_labels_unique:
                if cl == -1:
                    continue
                # get most frequent label
                # pred_under_cluster = list(pred[cond][cluster_labels == cl])
                pred_under_cluster = list(pred[cluster_labels == cl])
                freq_label = Counter(pred_under_cluster).most_common(1)[0][
                    0]  # max(set(pred_under_cluster), key=pred_under_cluster.count)
                pred_fixed[cluster_labels == cl] = freq_label

            new_pred = copy.deepcopy(pred)
            # new_pred[cond] = pred_fixed
            new_pred = pred_fixed
            metrics_dict, iou_classwise = self.compute_metrics(target, out=out, pred=new_pred, mode='eval')
        return metrics_dict, iou_classwise

    @torch.no_grad()
    def eval(self, loader, loss_fn=None, load_from_path=None, mode='val', epoch=0, ignored_labels=None):
        if load_from_path:
            self.model.load_state_dict(torch.load(load_from_path)['model_state_dict'])
        self.model.eval()
        metrics_dict_all, metrics_dict_all_post, iou_classwise_all, iou_classwise_all_post = {}, {}, {}, {}
        cm = ConfusionMatrix(self.config.n_classes)
        for i, data in enumerate(loader):
            # if i+1 in [17, 18, 21, 25, 44, 46]:
            step = i + len(loader) * epoch
            data = data.to(self.device)
            out = self.model(data)
            out = out.cpu()
            target = torch.squeeze(data.y).type(torch.LongTensor).cpu()
            pred = np.argmax(out, axis=-1)
            road_idxs = np.where(pred == 1)  # 1 - label for road, todo: remove this ugliness:)

            metrics_dict, iou_classwise = self.compute_metrics(target=target, out=out, pred=pred,
                                                               loss_fn=loss_fn, mode=mode)

            metrics_dict_all = {key: metrics_dict.get(key, []) + metrics_dict_all.get(key, [])
                                for key in set(list(metrics_dict.keys()) + list(metrics_dict_all.keys()))}
            iou_classwise_all = {key: iou_classwise.get(key, []) + iou_classwise_all.get(key, [])
                                 for key in set(list(iou_classwise.keys()) + list(iou_classwise_all.keys()))}
            # self.print_res(iou_classwise_all, "Before clustering:", False)

            accuracy = metrics_dict["accuracy"][0]
            loss = metrics_dict['loss'][0]
            print(f'[{i + 1}/{len(loader)}]'
                  f'Eval Acc: {accuracy:.4f}')
            self.print_res(iou_classwise, title='Classwise NN results for sample:', classwise=True,
                           mean_over_nonzero=False)

            # mask = ~np.in1d(target, self.ignore_label)
            # cm.count_predicted_batch(ground_truth_vec=target[mask], predicted=pred[mask])
            cm.count_predicted_batch(ground_truth_vec=target, predicted=pred)

            if mode != 'val':
                # if i+1 in [18, 21, 24, 25, 44, 46]:
                # if i > 22:
                # pcd = torchdata2o3dpcd(data.cpu(), visualise=True)
                # if self.config.mode == 1:
                #     pcd = pcd.select_by_index(np.asarray(road_idxs).squeeze(0))
                # print("TARGET CLASSES")
                # print(set(np.asarray(target.cpu())))
                # pred_remapped = self.remap_label_for_drawing(pred)
                # draw_pc_with_labels(data.pos.cpu(), np.asarray(pred_remapped), len(set(pred_remapped)), title="Predictions")
                #
                # cluster_labels, clusters = cluster_with_intensities(pcd, mode=self.config.mode, visualise=True, do_pca=True)

                if self.config.eval_clustering:
                    print('Clustering is being done..')
                    metrics_dict, iou_classwise = self.cluster_and_apply_frequent_label(batch_data=data, out=out, target=target)
                    if metrics_dict is not None:
                        metrics_dict_all_post = {key: metrics_dict.get(key, []) + metrics_dict_all_post.get(key, [])
                                                 for key in
                                                 set(list(metrics_dict.keys()) + list(metrics_dict_all_post.keys()))}
                        iou_classwise_all_post = {key: iou_classwise.get(key, []) + iou_classwise_all_post.get(key, [])
                                                  for key in
                                                  set(list(iou_classwise.keys()) + list(iou_classwise_all_post.keys()))}

                # new_pred_after_clustering = self.compute_under_clusters(cluster_labels, clusters, pred)
                # pred_remapped = self.remap_label_for_drawing(new_pred_after_clustering)
                #
                # metrics_dict_c, iou_classwise_c = self.compute_metrics(target, out=out, pred=new_pred_after_clustering, mode='eval')
                # self.print_res(iou_classwise_c, title='Classwise NN+CLUSTERS results for sample:', classwise=True)
                # # draw_pc_with_labels(data.pos.cpu(), np.asarray(pred_remapped), len(set(pred_remapped)), title="Predictions after adjustments")
                #
                # metrics_dict_all_post = {key: metrics_dict_c.get(key, []) + metrics_dict_all_post.get(key, [])
                #                          for key in
                #                          set(list(metrics_dict_c.keys()) + list(metrics_dict_all_post.keys()))}
                # iou_classwise_all_post = {key: iou_classwise_c.get(key, []) + iou_classwise_all_post.get(key, [])
                #                           for key in
                #                           set(list(iou_classwise_c.keys()) + list(iou_classwise_all_post.keys()))}

                # if config.verbose and (i + 1) % log_img_every == 0:

                # target_remapped = self.remap_label_for_drawing(data.y.cpu())
                # wandb.log(
                #     {
                #         "val_loss": loss,
                #         "val_acc": accuracy,
                #         "val_iteration": step
                #         # 'eval_inputs': wandb.Object3D(
                #         #     np.column_stack((np.array(data.pos.cpu()), np.array(data.x.cpu()) * 255))),
                #         # 'eval_targets': wandb.Object3D(
                #         #     np.column_stack((np.array(data.pos.cpu()), target_remapped))),
                #         # 'eval_predictions': wandb.Object3D(
                #         #     np.column_stack((np.array(data.pos.cpu()), pred_remapped)))
                #         # 'clusters': wandb.Object3D(
                #         #     np.column_stack((np.array(xyz_filtered.cpu()), cluster_labels + 1)))
                #         # 'eval_accuracy': acc,
                #         # 'miou_micro': iou_micro, 'miou_weighted': iou_weighted, 'miou_macro': iou_macro
                #     })
            # else:
            #     wandb.log({"val_loss": loss,
            #                "val_acc": accuracy,
            #                "val_iteration": step
            #                # 'eval_inputs': wandb.Object3D(
            #                #     np.column_stack((np.array(data.pos.cpu()), np.array(data.x[:, :3].cpu()) * 255)))
            #                })

        print(f'mIoU (based on conf. matrix): {cm.get_average_intersection_union()}')
        print(f'mAcc (based on conf. matrix): {cm.get_overall_accuracy()}')
        print(cm.get_intersection_union_per_class())
        print(cm.get_mean_class_accuracy())

        if mode != 'val':
            self.print_res(metrics_dict_all_post, title='ALL METRICS NN', print_overall_mean=False,
                           mean_over_nonzero=False)
            self.print_res(iou_classwise_all, title='Classwise NN results:', classwise=True, mean_over_nonzero=False)
            if self.config.eval_clustering:
                self.print_res(iou_classwise_all_post, title='After clustering:', classwise=True)
                self.print_res(metrics_dict_all_post, title='ALL METRICS (after clustering)', print_overall_mean=False)
        return metrics_dict_all, metrics_dict_all_post
