import torch
import torch.nn.functional as F
import wandb
import numpy as np
from collections import defaultdict
import open3d as o3d
import copy
from collections import Counter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from torchgeometry.losses import FocalLoss

from dl_task import DLTask
from metrics.confusion_matrix import ConfusionMatrix
from data.pcd_utils import torchdata2o3dpcd, cluster_with_intensities, draw_pc_with_labels, minmax, rgb2gray, center, \
    dbscan_cluster_sklearn, get_accuracy, compute_metrics, normalise_to_main_color


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
            accuracy = get_accuracy(out, target)
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

    def dbscan_cluster_o3d(self, xyz, rgb, eps=0.01, min_points=4):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
        pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        cluster_labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        return cluster_labels

    def dbscan_cluster_sklearn_rgb(self, rgb, xyz=None, eps=0.014, min_points=20):
        # grey_colors = np.asarray(rgb2gray(rgb * 255))
        # grey_colors = minmax(center(grey_colors))
        # intensity_variance = np.var(grey_colors)
        # avg_var = np.mean(intensity_variance)
        # print(f'DIST VAR: {avg_var}')
        rgb = np.asarray(rgb)
        X = np.column_stack((rgb, xyz))  # grey_colors.reshape(-1, 1)
        eps = 0.03
        # eps = 0.3  # 10, 9 is nice with min points=20, 17
        db = DBSCAN(eps=eps, min_samples=30).fit(X)
        return db.labels_

    def compose_condition(self, a, vals):
        cond = (a != vals[0])
        for i in range(1, len(vals)):
            cond &= (a != vals[i])
        return cond

    def draw_pc_with_labels(self, xyz, labels, window_name="PC with labels"):
        max_label = labels.max()
        # print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels % 20)  # (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0

        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

    def draw_pc_with_colors(self, xyz, colors, window_name="PC with colors"):
        pcd = o3d.geometry.PointCloud()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries([pcd], window_name=window_name)

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

    def cluster_and_apply_frequent_label(self, batch_data, out, target, visualise=False):
        xyz = batch_data.pos.cpu()
        rgb = batch_data.x[:, :3].cpu()  # todo doesnt look right
        pred = out.cpu()  # np.argmax(out, axis=-1) - for pointnet realtime preds todo!

        self.draw_pc_with_colors(xyz, rgb)
        # self.draw_pc_with_colors(xyz, normalise_to_main_color(rgb))

        rgb = normalise_to_main_color(rgb)

        classes_to_filter_out = [21, 11, 22]  # 7, 8, 12, 13]  # todo move this into config

        cond = self.compose_condition(pred, classes_to_filter_out)

        n_conditions = len(classes_to_filter_out)
        xyz_filtered = xyz[cond] if n_conditions > 0 else xyz
        pred_filtered = pred[cond] if n_conditions > 0 else pred
        rgb_filtered = rgb[cond] if n_conditions > 0 else rgb
        target_f = target[cond] if n_conditions > 0 else target
        # cluster_labels = self.dbscan_cluster_o3d(xyz_filtered.cpu(), rgb_filtered.cpu(), eps=0.025,
        #                                          min_points=50)  # not bad with 0,03 -> for pole +2%

        metrics_dict, iou_classwise = None, None
        # if len(xyz_filtered) > 0:
        #     cluster_labels = dbscan_cluster_sklearn(xyz_filtered, rgb_filtered, eps=self.config.clustering_eps,
        #                                             min_points=self.config.clustering_min_points)
        #
        #     if visualise:
        #         print(f"classes in pred: {set(np.asarray(pred_filtered))}")
        #         self.draw_pc_with_labels(xyz_filtered, target_f, window_name="Target")
        #         self.draw_pc_with_labels(xyz_filtered, pred_filtered, window_name="Predictions")
        #         self.draw_pc_with_labels(xyz_filtered, cluster_labels, window_name="Clusters")
        #
        #     # iterate through clusters
        #     cluster_labels_unique = np.unique(cluster_labels)
        #     print(f'Clusters cnt: {cluster_labels_unique}')
        #     pred_fixed = copy.deepcopy(pred_filtered)
        #     for cl in cluster_labels_unique:
        #         if cl == -1:
        #             continue
        #
        #         rgb2 = rgb_filtered[cluster_labels == cl]
        #         xyz2 = xyz_filtered[cluster_labels == cl]
        #         cluster_labels_c = self.dbscan_cluster_sklearn_rgb(rgb2, xyz2)
        #         cluster_labels_unique_c = np.unique(cluster_labels_c)
        #         print(f'Clusters cnt RGB: {cluster_labels_unique_c}')
        #         self.draw_pc_with_colors(xyz2, rgb2)
        #         self.draw_pc_with_labels(xyz2, cluster_labels_c, window_name="Clusters RGB")
        #
        #         # get most frequent label
        #         pred_under_cluster = np.asarray(list(pred_filtered[cluster_labels == cl]))
        #         freq_label = Counter(pred_under_cluster).most_common(1)[0][0]
        #         # if len(pred_under_cluster[pred_under_cluster == int(freq_label)]) / len(pred_under_cluster) > 0.85:
        #         pred_fixed[cluster_labels == cl] = freq_label
        #
        #     new_pred = copy.deepcopy(pred)
        #     if n_conditions > 0:
        #         new_pred[cond] = pred_fixed
        #     else:
        #         new_pred = pred_fixed
        #     metrics_dict, iou_classwise = compute_metrics(target=target, out=new_pred, pred=new_pred, mode='eval')
        return metrics_dict, iou_classwise

    @torch.no_grad()
    def eval(self, loader, loss_fn=None, load_from_path=None, mode='val', epoch=0, ignored_labels=None):
        if load_from_path:
            self.model.load_state_dict(torch.load(load_from_path)['model_state_dict'])
        self.model.eval()
        metrics_dict_all, metrics_dict_all_post, iou_classwise_all, iou_classwise_all_post = {}, {}, {}, {}
        # m = ConfusionMatrix(self.config.n_classes)
        for i, data in enumerate(loader):
            step = i + len(loader) * epoch
            data = data.cpu()
            out = self.model(data)
            out = out.cpu()
            target = torch.squeeze(data.y[:, 0]).type(torch.LongTensor).cpu()
            pred = data.y[:, 1].cpu().int()  # np.argmax(out, axis=-1)
            # road_idxs = np.where(pred == 1)  # 1 - label for road, todo: remove this ugliness:)

            # self.draw_pc_with_labels(np.asarray(data.pos), np.asarray(target), window_name="GT")
            # self.draw_pc_with_labels(np.asarray(data.pos), np.asarray(pred), window_name="Prediction")

            # out = pred
            metrics_dict, iou_classwise = compute_metrics(target=target, out=out, pred=pred,
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
            self.print_res(iou_classwise, title='Classwise NN results for sample:', classwise=False,
                           mean_over_nonzero=False)

            # mask = ~np.in1d(target, self.ignore_label)
            # cm.count_predicted_batch(ground_truth_vec=target[mask], predicted=pred[mask])
            # cm.count_predicted_batch(ground_truth_vec=target, predicted=pred)

            if mode != 'val':
                # if i+1 in [18, 21, 24, 25, 44, 46]:
                # if i > 22:
                # pcd = torchdata2o3dpcd(data.cpu(), visualise=False)
                # if self.config.mode == 1:
                #     pcd = pcd.select_by_index(np.asarray(road_idxs).squeeze(0))
                # print("TARGET CLASSES")
                # print(set(np.asarray(target.cpu())))
                # pred_remapped = self.remap_label_for_drawing(pred)
                # draw_pc_with_labels(data.pos.cpu(), np.asarray(pred), len(set(pred)), title="Predictions")

                # metrics_dict_c, iou_classwise_c = cluster_with_intensities(pcd, pred=pred, target=target, mode=self.config.mode, visualise=False, do_pca=True)

                if self.config.eval_clustering:
                    print('Clustering is being done..')
                    metrics_dict, iou_classwise = self.cluster_and_apply_frequent_label(batch_data=data, out=out,
                                                                                        target=target)
                    if metrics_dict is not None:
                        metrics_dict_all_post = {key: metrics_dict.get(key, []) + metrics_dict_all_post.get(key, [])
                                                 for key in
                                                 set(list(metrics_dict.keys()) + list(metrics_dict_all_post.keys()))}
                        iou_classwise_all_post = {key: iou_classwise.get(key, []) + iou_classwise_all_post.get(key, [])
                                                  for key in
                                                  set(list(iou_classwise.keys()) + list(iou_classwise_all_post.keys()))}
                        self.print_res(iou_classwise, title='Classwise NN+Clustering results for sample:',
                                       classwise=False,
                                       mean_over_nonzero=False)

                # draw_pc_with_labels(data.pos.cpu(), np.asarray(pred_remapped), len(set(pred_remapped)), title="Predictions after adjustments")

                # metrics_dict_all_post = {key: metrics_dict_c.get(key, []) + metrics_dict_all_post.get(key, [])
                #                          for key in
                #                          set(list(metrics_dict_c.keys()) + list(metrics_dict_all_post.keys()))}
                # iou_classwise_all_post = {key: iou_classwise_c.get(key, []) + iou_classwise_all_post.get(key, [])
                #                           for key in
                #                           set(list(iou_classwise_c.keys()) + list(iou_classwise_all_post.keys()))}

                # if config.verbose and (i + 1) % log_img_every == 0:

                # target_remapped = self.remap_label_for_drawing(data.y.cpu())
                wandb.log(
                    {
                        "val_loss": loss,
                        "val_acc": accuracy,
                        "val_iteration": step
                        # 'eval_inputs': wandb.Object3D(
                        #     np.column_stack((np.array(data.pos.cpu()), np.array(data.x.cpu()) * 255))),
                        # 'eval_targets': wandb.Object3D(
                        #     np.column_stack((np.array(data.pos.cpu()), target_remapped))),
                        # 'eval_predictions': wandb.Object3D(
                        #     np.column_stack((np.array(data.pos.cpu()), pred_remapped)))
                        # 'clusters': wandb.Object3D(
                        #     np.column_stack((np.array(xyz_filtered.cpu()), cluster_labels + 1)))
                        # 'eval_accuracy': acc,
                        # 'miou_micro': iou_micro, 'miou_weighted': iou_weighted, 'miou_macro': iou_macro
                    })
            # else:
            #     wandb.log({"val_loss": loss,
            #                "val_acc": accuracy,
            #                "val_iteration": step
            #                # 'eval_inputs': wandb.Object3D(
            #                #     np.column_stack((np.array(data.pos.cpu()), np.array(data.x[:, :3].cpu()) * 255)))
            #                })

        # print(f'mIoU (based on conf. matrix): {cm.get_average_intersection_union()}')
        # print(f'mAcc (based on conf. matrix): {cm.get_overall_accuracy()}')
        # print(cm.get_intersection_union_per_class())
        # print(cm.get_mean_class_accuracy())

        if mode != 'val':
            self.print_res(metrics_dict_all, title='ALL METRICS NN', print_overall_mean=False,
                           mean_over_nonzero=False)
            self.print_res(iou_classwise_all, title='Classwise NN results:', classwise=False, mean_over_nonzero=False)
            if self.config.eval_clustering:
                self.print_res(iou_classwise_all_post, title='After clustering:', classwise=False,
                               mean_over_nonzero=False)
                self.print_res(metrics_dict_all_post, title='ALL METRICS (after clustering)', print_overall_mean=False,
                               mean_over_nonzero=False)
        return metrics_dict_all, metrics_dict_all_post
