import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score

matplotlib.use('TkAgg')

from plyfile import PlyData, PlyElement


def draw_pc_with_labels(xyz, labels, num_clusters, title=""):
    colors = plt.cm.get_cmap("jet")(labels / max(labels)) if num_clusters > 20 else plt.cm.get_cmap("tab20")(
        labels % 20)

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.visualization.draw_geometries([pcd], width=1200, height=1000, window_name=title)


def center(arr):
    return arr - arr.mean()


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def rgb2gray(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def dbscan_cluster_sklearn(features, eps=0.014, min_points=20):
    print("CLUSTERING...")
    db = DBSCAN(eps=eps, min_samples=min_points).fit(features)
    return db.labels_ + 1


class LineMarkingDetector:
    def __init__(self, visualise=True, do_pca=True, eps=0.5, min_points=20, cluster_on_grey=False,
                 cluster_on_xyz=False, cluster_on_rgb=False, use_true_colors_for_visualisation=False):

        self.visualise = visualise
        self.do_pca = do_pca
        self.cluster_on_grey = cluster_on_grey
        self.cluster_on_xyz = cluster_on_xyz
        self.use_true_colors = use_true_colors_for_visualisation
        self.eps = eps
        self.min_points = min_points
        self.cluster_on_rgb = cluster_on_rgb

    def prep_and_cluster(self, pcd):
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
        normals = np.asarray(pcd.normals)
        colors = np.asarray(pcd.colors)

        xyz = minmax(center(np.asarray(pcd.points)))
        if self.cluster_on_grey and self.cluster_on_xyz:
            print("Clustering on xyz and grey")
            grey_colors = rgb2gray(colors)
            grey_colors = minmax(center(grey_colors))
            plt.hist(grey_colors)
            plt.show()
            grey_colors = np.array([0 if i < 0.7 else i for i in grey_colors])
            cl_labels = dbscan_cluster_sklearn(np.column_stack((xyz, grey_colors)), eps=self.eps,
                                               min_points=self.min_points)
        elif self.cluster_on_xyz:
            print("Clustering on xyz only")
            cl_labels = dbscan_cluster_sklearn(xyz)
        elif self.cluster_on_grey:
            print("Clustering on grey")
            grey_colors = rgb2gray(colors)
            grey_colors = minmax(center(grey_colors))
            plt.hist(grey_colors)
            plt.show()
            grey_colors = np.array([0 if i < 0.7 else i for i in grey_colors])
            cl_labels = dbscan_cluster_sklearn(grey_colors.reshape(1, -1), eps=self.eps,
                                               min_points=self.min_points)
        elif self.cluster_on_rgb:
            cl_labels = dbscan_cluster_sklearn(colors, eps=self.eps,
                                               min_points=self.min_points)
        else:
            cl_labels = None
            print("SET WHAT TO CLUSTER")
        clusters = set(cl_labels)
        print(f"Num of clusters {len(clusters)}")
        print(clusters)
        if self.visualise:
            draw_pc_with_labels(xyz, cl_labels, num_clusters=len(clusters), title="Found clusters")
        return clusters, cl_labels, normals, xyz, colors

    def prep_o3d_pc(self, points, normals=None, pc_colors=None, uniform_color=None):
        if uniform_color is None:
            uniform_color = [0, 0, 1]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals != None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if self.use_true_colors:
            if pc_colors is None:
                print("Please specify pc colors")
            pcd.colors = o3d.utility.Vector3dVector(pc_colors)
        else:
            pcd.paint_uniform_color(uniform_color)
        return pcd

    def mesh_markings(self, pcd):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=8)
        bbox = pcd.get_oriented_bounding_box()
        mesh = mesh.crop(bbox)
        return mesh

    def mesh_road_surfaces(self, pcd):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, depth=15)
        densities = np.asarray(densities)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        bbox = pcd.get_oriented_bounding_box()
        mesh = mesh.crop(bbox)
        return mesh

    def cluster_road_surfaces(self, pcd):
        clusters, labels, normals, xyz, colors = self.prep_and_cluster(
            pcd)  # 0.07 for 50k density on 10 cuts, 0.04 - for 100k on 10 cuts)
        markings, roads, meshes = list(), list(), list()

        if self.do_pca:
            pca = PCA(n_components=2)
            clusters.remove(0)  # remove noise cluster
            cluster_labels_and_size_dict = dict()
            for c in clusters:
                cluster_idx = np.where(labels == c)
                cluster_points = xyz[cluster_idx]
                cluster_labels_and_size_dict[c] = len(cluster_points)

                # Fit PCA
                pca.fit_transform(cluster_points)
                pca_variances = pca.explained_variance_ratio_

                pcd = self.prep_o3d_pc(points=cluster_points, normals=normals[cluster_idx],
                                       pc_colors=colors[cluster_idx], uniform_color=[0, 0, 1])

                if pca_variances[0] - pca_variances[1] >= 0.9:
                    markings.append(pcd)
                else:
                    if self.use_true_colors:
                        pcd.colors = o3d.utility.Vector3dVector(colors[cluster_idx])
                    else:
                        pcd.paint_uniform_color([0.5, 0.5, 0.5])
                    roads.append(pcd)

            cluster_labels_and_size_dict = dict(
                sorted(cluster_labels_and_size_dict.items(), key=lambda item: item[1], reverse=True))
            # add road surface
            cluster_idx = np.where(labels == list(cluster_labels_and_size_dict)[0])
            pcd = self.prep_o3d_pc(points=xyz[cluster_idx], normals=normals[cluster_idx],
                                   pc_colors=colors[cluster_idx], uniform_color=[0.5, 0.5, 0.5])
            roads.append(pcd)

            if self.visualise:
                o3d.visualization.draw_geometries(roads + markings, width=1200, height=1000,
                                                  window_name="Road markings and road surface")

        return labels, clusters


# pcd = o3d.io.read_point_cloud('./HE-PHASE-2_A11_570687_271766 WAN - Cloud_cut1.ply')
# plydata = PlyData.read('./HE-PHASE-2_A11_570687_271766 WAN - Cloud_cut1 - Cloud_cut2.ply')
# plydata = PlyData.read('./HE-PHASE-2_A11_570687_271766 WAN - Cloud_cut1.ply')

pcd = o3d.io.read_point_cloud('./HE-PHASE-2_A11_570332_271173 wan - Cloud.ply')
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=pcd.get_center())

# o3d.visualization.draw_geometries([mesh_frame, pcd])

plydata = PlyData.read('./HE-PHASE-2_A11_570332_271173 wan - Cloud.ply')


# property double x
# property double y
# property double z
# property uchar red
# property uchar green
# property uchar blue
# property float scalar_PointSourceId
# property float scalar_UserData
# property float scalar_GpsTime
# property float scalar_Intensity
# property float scalar_Classification


def normalise(arr):
    return (arr - arr.mean(axis=0)) / np.max(np.abs((arr - arr.mean(axis=0))))


z = plydata.elements[0].data['z']
# separate ground by mean np.mean(z)

ground_idx = np.where(z < np.mean(z))
z = z[ground_idx]

x = plydata.elements[0].data['x'][ground_idx]
y = plydata.elements[0].data['y'][ground_idx]

r = plydata.elements[0].data['red'][ground_idx]
g = plydata.elements[0].data['green'][ground_idx]
b = plydata.elements[0].data['blue'][ground_idx]
rgb = np.column_stack((r, g, b))
intensities = plydata.elements[0].data['scalar_Intensity'][ground_idx]
gts = plydata.elements[0].data['scalar_Classification'][ground_idx]

xyz = normalise(np.column_stack((x, y, z)))
intensities = normalise(intensities)

# plt.hist(intensities)
# plt.show()

lightest_idx = np.where(intensities > 0.2)
# intensities = np.asarray([1 if i > 0.2 else i for i in intensities])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz[lightest_idx])
pcd.paint_uniform_color([1.0, 0, 0])
# pcd.colors = o3d.utility.Vector3dVector(np.column_stack((r[lightest_idx], g[lightest_idx], b[lightest_idx])))
# o3d.visualization.draw_geometries([pcd])

intensities = intensities[lightest_idx]

# cl_labels = dbscan_cluster_sklearn(np.column_stack((xyz, intensities)), eps=0.05, min_points=50)
# cl_labels = dbscan_cluster_sklearn(intensities.reshape(-1, 1), eps=0.05, min_points=50)
cl_labels = dbscan_cluster_sklearn(xyz[lightest_idx], eps=0.005, min_points=30)

clusters = set(cl_labels)
print(f"Num of clusters {len(clusters)}")
print(clusters)

no_noise_cluster_idx = np.where(cl_labels != 0)

cl_labels_no_noise = cl_labels[no_noise_cluster_idx]
points_no_noiise = xyz[lightest_idx][no_noise_cluster_idx]
# draw_pc_with_labels(points_no_noiise, cl_labels_no_noise, num_clusters=len(clusters), title="Found clusters")


pca = PCA(n_components=2)
clusters.remove(0)  # remove noise cluster
cluster_labels_and_size_dict = dict()
markings = list()
marking_idx = []
for c in clusters:
    cluster_idx = np.where(cl_labels == c)
    cluster_points = xyz[lightest_idx][cluster_idx]
    cluster_labels_and_size_dict[c] = len(cluster_points)

    # Fit PCA
    pca.fit_transform(cluster_points)
    pca_variances = pca.explained_variance_ratio_

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    pcd.paint_uniform_color([0, 0, 1])

    if pca_variances[0] - pca_variances[1] >= 0.9:
        markings.append(pcd)
        for i in cluster_idx:
            marking_idx.append(i)
#
# o3d.visualization.draw_geometries(markings, width=1200, height=1000,
#                                                   window_name="Road markings and road surface")

pred_line_marking = np.zeros(len(xyz[lightest_idx]))

for i in marking_idx:
    pred_line_marking[i] = 1

gts = [1 if i == 9 else 0 for i in gts[lightest_idx]]

iou = jaccard_score(y_true=gts, y_pred=pred_line_marking, average='binary')

print("RES")
print(iou)

# o3d.visualization.draw_geometries([pcd])
# detector = LineMarkingDetector(eps=0.03, min_points=100, cluster_on_xyz=True, cluster_on_grey=True)
# detector = LineMarkingDetector(eps=0.0001, min_points=50, cluster_on_xyz=False, cluster_on_grey=True)
# detector = LineMarkingDetector(eps=0.0001, min_points=50, cluster_on_rgb=True)
# detector.cluster_road_surfaces(pcd)
