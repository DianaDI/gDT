import glob
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import json
from collections import defaultdict
from os.path import basename
import open3d as o3d
from sklearn.cluster import DBSCAN
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score


def get_accuracy(out, target):
    correct_nodes = out.eq(target).sum().item()  # out.argmax(dim=1).eq(target).sum().item() todo fix this
    return correct_nodes / len(target)


def get_nearest_point(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


def compute_metrics(target, out, pred, loss_fn=None, mode="val"):
    metrics_dict = defaultdict(list)
    iou_classwise = defaultdict(list)

    # loss = F.nll_loss(out, target) if not loss_fn else loss_fn(out.t().unsqueeze(0).unsqueeze(2),
    #                                                            target.unsqueeze(0).unsqueeze(1))
    # metrics_dict['loss'].append(loss)
    acc = get_accuracy(out, target)
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


def get_trajectory_points(pcd_path, poses):
    ranges = basename(pcd_path).split(".")[0].split("_")
    start, end = int(ranges[0]), int(ranges[1])
    points = []
    for line in poses:
        l_parts = line.split(" ")
        if start <= int(l_parts[0]) <= end:
            x, y, z = l_parts[4], l_parts[8], l_parts[-1]
            points.append(np.asarray((float(x), float(y), float(z))))
    return np.asarray(points)


def cut_with_trajectory(n, pcd_path, traj_poses, xyz, rgb, labels, predictions=None):
    traj_points = get_trajectory_points(pcd_path, traj_poses)
    part_len = int(len(traj_points) / n)
    split_traj_points = [traj_points[i] for i in range(part_len, len(traj_points), part_len)]

    parts_dict_xyz, parts_dict_rgb, parts_dict_lbl, parts_dict_pred = defaultdict(list), defaultdict(list), defaultdict(
        list), defaultdict(list)

    for i in range(len(xyz)):
        p, r, l = xyz[i], rgb[i], labels[i]
        pred = None if predictions is None else predictions[i]
        nearest = get_nearest_point(p, split_traj_points)
        parts_dict_xyz[nearest].append(p)
        parts_dict_rgb[nearest].append(r)
        parts_dict_lbl[nearest].append(l)
        if predictions is not None:
            parts_dict_pred[nearest].append(pred)

    split_parts = []

    for i in range(len(split_traj_points)):
        arr_part = np.column_stack(
            (parts_dict_xyz[i], parts_dict_rgb[i], parts_dict_lbl[i])) if predictions is None else np.column_stack(
            (parts_dict_xyz[i], parts_dict_rgb[i], parts_dict_lbl[i], parts_dict_pred[i]))
        split_parts.append(arr_part)

    return split_parts


def cut_boxes(arr, n):
    # decide whether to cut along x or y
    x_len = max(arr[:, 0]) - min(arr[:, 0])
    y_len = max(arr[:, 1]) - min(arr[:, 1])
    axis = 0 if x_len >= y_len else 1

    ax = arr[:, axis]
    part_len = (np.max(ax) - np.min(ax)) / n
    arrs = list()
    for i in range(n):
        if i != n - 1:
            mask = np.logical_and(arr[:, axis] > np.min(ax) + part_len * i,
                                  arr[:, axis] < np.min(ax) + part_len * (i + 1))
            arr_temp = arr[mask]
        else:
            arr_temp = arr[arr[:, axis] > np.min(ax) + part_len * i]
        arrs.append(arr_temp)
    return arrs


def read_fields(path, xyz=True, rgb=True, label=True):
    pcd = PlyData.read(path)
    pcdv = pcd['vertex']
    XYZ = np.column_stack((pcdv['x'], pcdv['y'], pcdv['z'])) if xyz else None
    RGB = np.column_stack((pcdv['red'] / 255, pcdv['green'] / 255, pcdv['blue'] / 255)) if rgb else None
    labels = np.array(pcdv['semantic']) if label else None
    return XYZ, RGB, labels


def count_class_labels():
    all_files = glob.glob("C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/*_sync/static/*.ply")
    counts = {}
    for file in tqdm(all_files):
        _, _, labels = read_fields(file, xyz=False, rgb=False, label=True)
        for l in labels:
            l = int(l)
            if l in counts.keys():
                counts[l] += 1
            else:
                counts[l] = 1
    print(counts)
    with open("class_label_counts.json", 'w') as fp:
        json.dump(counts, fp)


def rgb2gray(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def change_background_to_black(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    return False


def torchdata2o3dpcd(data, colors=True, visualise=True, gray=False):
    pcd = o3d.geometry.PointCloud()
    xyz = np.array(data.pos)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(data.x[:, :3])
    if gray:
        gray = rgb2gray(data.x[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack((gray, gray, gray)))
    if visualise:
        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)
    return pcd


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def center(arr):
    return arr - arr.mean()


def dbscan_cluster_sklearn(xyz=None, rgb=None, eps=0.014, min_points=20):
    # X = np.column_stack((xyz, rgb))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f'DIST AVG: {avg_dist}')
    X = xyz
    eps = avg_dist * 8  # 10, 9 is nice with min points=20, 17
    db = DBSCAN(eps=eps, min_samples=10).fit(X)
    return db.labels_


def draw_pc_with_labels(xyz, labels, num_clusters, title=""):
    colors = plt.cm.get_cmap("jet")(labels / max(labels)) if num_clusters > 20 else plt.cm.get_cmap("tab20")(
        labels % 20)

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd], width=1200, height=1000, window_name=title)


def prep_and_cluster(pcd, eps, min_points=20, visualise=False, cluster_on_grey=True, cluster_on_xyz=True):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors)
    # grey_filtered = np.array([0 if i < 0.2 else i for i in grey_colors])

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f'DIST AVG: {avg_dist}')

    xyz = np.asarray(pcd.points)
    if cluster_on_grey and cluster_on_xyz:
        print("Clustering on xyz and grey")
        grey_colors = rgb2gray(colors)
        grey_colors = minmax(center(grey_colors))
        intensity_variance = np.var(grey_colors)
        avg_var = np.mean(intensity_variance)
        print(f'VAR AVG: {avg_var}')
        cl_labels = dbscan_cluster_sklearn(np.column_stack((xyz, grey_colors)), eps=avg_dist * 10 + avg_var,
                                           min_points=min_points)  # =20)  # eps 06 finds very good line markings, 0.08 good for ground on 50k for 10 cuts scheme
    elif cluster_on_xyz:
        print("Clustering on xyz only")
        cl_labels = dbscan_cluster_sklearn(xyz)
    else:
        cl_labels = None
        print("SET WHAT TO CLUSTER")
    clusters = set(cl_labels)
    print(f"Num of clusters {len(clusters)}")
    print(clusters)
    if visualise:
        draw_pc_with_labels(xyz, cl_labels, num_clusters=len(clusters), title="Clusters")
    return clusters, cl_labels, normals, xyz, colors


def cluster_with_intensities_road_surfaces(pcd, visualise=False, do_pca=False, cluster_on_grey=True,
                                           cluster_on_xyz=True, use_true_colors=False):
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # print(f'DIST AVG: {avg_dist}')
    clusters, labels, normals, xyz, colors = prep_and_cluster(pcd, eps=0.03, min_points=20, visualise=visualise,
                                                              # 0.07 for 50k density on 10 cuts, 0.04 - for 100k on 10 cuts
                                                              cluster_on_grey=cluster_on_grey,
                                                              cluster_on_xyz=cluster_on_xyz)

    markings = list()
    roads = list()
    meshes = list()

    if do_pca:
        pca = PCA(n_components=2)

        clusters.remove(0)  # remove noise cluster
        cluster_labels_and_size_dict = dict()
        for c in clusters:
            cluster_idx = np.where(labels == c)
            cluster_points = xyz[cluster_idx]
            cluster_labels_and_size_dict[c] = len(cluster_points)

            pca.fit_transform(cluster_points)
            pca_variances = pca.explained_variance_ratio_

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cluster_points)
            pcd.normals = o3d.utility.Vector3dVector(normals[cluster_idx])
            if use_true_colors:
                pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
            else:
                pcd.paint_uniform_color([0, 0, 1])

            if pca_variances[0] - pca_variances[1] >= 0.8:
                markings.append(pcd)
            else:
                # pcd.paint_uniform_color([0.5, 0.5, 0.5])
                pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
                roads.append(pcd)

        cluster_labels_and_size_dict = dict(
            sorted(cluster_labels_and_size_dict.items(), key=lambda item: item[1], reverse=True))
        # add road surface
        cluster_idx = np.where(labels == list(cluster_labels_and_size_dict)[0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[cluster_idx])
        pcd.normals = o3d.utility.Vector3dVector(normals[cluster_idx])
        # if use_true_colors:
        pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
        # else:
        #     pcd.paint_uniform_color([0.5, 0.5, 0.5])
        roads.append(pcd)

        cluster_idx = np.where(labels == list(cluster_labels_and_size_dict)[1])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[cluster_idx])
        pcd.normals = o3d.utility.Vector3dVector(normals[cluster_idx])
        # if use_true_colors:
        pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
        # else:
        #     pcd.paint_uniform_color([0.5, 0.5, 0.5])
        roads.append(pcd)

        if visualise:
            o3d.visualization.draw_geometries(roads + markings, width=1200, height=1000)

        for pc in tqdm(markings):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pc, depth=8)

            bbox = pc.get_oriented_bounding_box()
            mesh = mesh.crop(bbox)
            meshes.append(mesh)

        for pc in tqdm(roads):
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pc, depth=15)

            densities = np.asarray(densities)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)

            bbox = pc.get_oriented_bounding_box()
            mesh = mesh.crop(bbox)
            meshes.append(mesh)

        o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True, width=1200, height=1000)

    return labels, clusters


def cluster_with_intensities_above_road(pcd, pred=None, target=None, visualise=False, do_pca=False,
                                        cluster_on_grey=False, cluster_on_xyz=True,
                                        use_true_colors=False):
    clusters, cl_labels, normals, xyz, colors = prep_and_cluster(pcd, eps=0.015, min_points=10, visualise=visualise,
                                                                 cluster_on_grey=cluster_on_grey,
                                                                 cluster_on_xyz=cluster_on_xyz)

    polelikes = list()
    meshes = list()

    new_pred_after_clustering = pred

    if do_pca:
        pca = PCA(n_components=2)

        clusters.remove(0)  # remove noise
        cluster_labels_and_size_dict = dict()
        for c in clusters:
            cluster_idx = np.where(cl_labels == c)
            cluster_points = xyz[cluster_idx]
            cluster_labels_and_size_dict[c] = len(cluster_points)

            pca.fit_transform(cluster_points)
            pca_variances = pca.explained_variance_ratio_

            if pca_variances[0] - pca_variances[1] >= 0.65:

                print(f'PCA: {pca_variances}')
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cluster_points)
                pcd.normals = o3d.utility.Vector3dVector(normals[cluster_idx])
                if use_true_colors:
                    pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
                else:
                    pcd.paint_uniform_color([0, 0, 1])

                bb = pcd.get_axis_aligned_bounding_box()
                x, y, z = bb.get_extent()
                print(f'xyz: {x} {y} {z}')

                # if visualise:
                #     o3d.visualization.draw_geometries([pcd])

                if z > x and z > y:
                    # get axis alligned bb and check dimensions. if z dim is longest then it is pole!
                    polelikes.append(pcd)
                    # apply label 17 (the only pole there) todo
                    new_pred_after_clustering[cluster_idx] = 17

        print(f"Pole-likes: {len(polelikes)}")
        # if visualise and len(polelikes) > 0:
        #     print("Pole like found")
        #     o3d.visualization.draw_geometries(polelikes, width=1200, height=1000)

        # pred_remapped = self.remap_label_for_drawing(new_pred_after_clustering)

    metrics_dict_c, iou_classwise_c = compute_metrics(target, out=new_pred_after_clustering, pred=new_pred_after_clustering,
                                                               mode='eval')

        # for pc in tqdm(polelikes):
        #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pc, depth=8)
        #     bbox = pc.get_oriented_bounding_box()
        #     mesh = mesh.crop(bbox)
        #     meshes.append(mesh)
        #
        # if visualise:
        #     o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True, width=1200, height=1000)

    return metrics_dict_c, iou_classwise_c


def cluster_with_intensities(pcd, mode, pred=None, target=None, visualise=False, do_pca=False):
    if mode == 1:
        return cluster_with_intensities_road_surfaces(pcd, visualise, do_pca, cluster_on_grey=True, cluster_on_xyz=True,
                                                      use_true_colors=False)
    elif mode == 2:
        return cluster_with_intensities_above_road(pcd=pcd, visualise=visualise, do_pca=do_pca, pred=pred, target=target,
                                                   cluster_on_grey=False, cluster_on_xyz=True)
    else:
        print("NOT IMPLEMENTED")


def normalise_to_main_color(rgb):
    rgb = rgb / np.asarray(rgb.max(dim=1).values)[:, None]
    # rgb = rgb / np.asarray(rgb.sum(dim=1))[:, None]
    return rgb
