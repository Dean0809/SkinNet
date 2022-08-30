import os
from sys import platform
import trimesh
import numpy as np
import open3d as o3d
import itertools as it

import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from utils import binvox_rw
from utils.rig_parser import Info
from utils.io_utils import assemble_skel_skin

from geometric_proc.common_ops import get_bones, calc_surface_geodesic
from geometric_proc.compute_volumetric_geodesic import pts2line, calc_pts2bone_visible_mat

from gen_dataset import get_tpl_edges, get_geo_edges

from main import post_filter

from models.skinnet import skinnet


def normalize_obj(mesh_v):
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, min(mesh_v[:, 1]),
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    return mesh_v, pivot, scale


def create_single_data(mesh_filaname):
    """
    create input data for the network. The data is wrapped by Data structure in pytorch-geometric library
    :param mesh_filaname: name of the input mesh
    :return: wrapped data, voxelized mesh, and geodesic distance matrix of all vertices
    """
    mesh = o3d.io.read_triangle_mesh(mesh_filaname)
    mesh.compute_vertex_normals()
    mesh_v = np.asarray(mesh.vertices)
    mesh_vn = np.asarray(mesh.vertex_normals)
    mesh_f = np.asarray(mesh.triangles)

    mesh_v, translation_normalize, scale_normalize = normalize_obj(mesh_v)
    mesh_normalized = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh_v), triangles=o3d.utility.Vector3iVector(mesh_f))
    o3d.io.write_triangle_mesh(mesh_filename.replace("_remesh.obj", "_normalized.obj"), mesh_normalized)

    # vertices
    v = np.concatenate((mesh_v, mesh_vn), axis=1)
    v = torch.from_numpy(v).float()

    # topology edges
    print("     gathering topological edges.")
    tpl_e = get_tpl_edges(mesh_v, mesh_f).T
    tpl_e = torch.from_numpy(tpl_e).long()
    tpl_e, _ = add_self_loops(tpl_e, num_nodes=v.size(0))

    # surface geodesic distance matrix
    print("     calculating surface geodesic matrix.")
    surface_geodesic = calc_surface_geodesic(mesh)

    # geodesic edges
    print("     gathering geodesic edges.")
    geo_e = get_geo_edges(surface_geodesic, mesh_v).T
    geo_e = torch.from_numpy(geo_e).long()
    geo_e, _ = add_self_loops(geo_e, num_nodes=v.size(0))

    # batch
    batch = torch.zeros(len(v), dtype=torch.long)

    # voxel
    if not os.path.exists(mesh_filaname.replace('_remesh.obj', '_normalized.binvox')):
        if platform == "linux" or platform == "linux2":
            os.system("./binvox -d 88 -pb " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        else:
            raise Exception('Sorry, we currently only support windows and linux.')

    data = Data(x=v[:, 3:6], pos=v[:, 0:3], tpl_edge_index=tpl_e, geo_edge_index=geo_e, batch=batch)
    return data, surface_geodesic, translation_normalize, scale_normalize


def calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=False):
    """
    calculate volumetric geodesic distance from vertices to each bones
    :param bones: B*6 numpy array where each row stores the starting and ending joint position of a bone
    :param mesh_v: V*3 mesh vertices
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: an approaximate volumetric geodesic distance matrix V*B, were (v,b) is the distance from vertex v to bone b
    """

    if subsampling:
        mesh0 = o3d.io.read_triangle_mesh(mesh_filename)
        mesh0 = mesh0.simplify_quadric_decimation(3000)
        o3d.io.write_triangle_mesh(mesh_filename.replace(".obj", "_simplified.obj"), mesh0)
        mesh_trimesh = trimesh.load(mesh_filename.replace(".obj", "_simplified.obj"))
        subsamples_ids = np.random.choice(len(mesh_v), np.min((len(mesh_v), 1500)), replace=False)
        subsamples = mesh_v[subsamples_ids, :]
        surface_geodesic = surface_geodesic[subsamples_ids, :][:, subsamples_ids]
    else:
        mesh_trimesh = trimesh.load(mesh_filename)
        subsamples = mesh_v
    origins, ends, pts_bone_dist = pts2line(subsamples, bones)
    pts_bone_visibility = calc_pts2bone_visible_mat(mesh_trimesh, origins, ends)
    pts_bone_visibility = pts_bone_visibility.reshape(len(bones), len(subsamples)).transpose()
    pts_bone_dist = pts_bone_dist.reshape(len(bones), len(subsamples)).transpose()
    # remove visible points which are too far
    for b in range(pts_bone_visibility.shape[1]):
        visible_pts = np.argwhere(pts_bone_visibility[:, b] == 1).squeeze(1)
        if len(visible_pts) == 0:
            continue
        threshold_b = np.percentile(pts_bone_dist[visible_pts, b], 15)
        pts_bone_visibility[pts_bone_dist[:, b] > 1.3 * threshold_b, b] = False

    visible_matrix = np.zeros(pts_bone_visibility.shape)
    visible_matrix[np.where(pts_bone_visibility == 1)] = pts_bone_dist[np.where(pts_bone_visibility == 1)]
    for c in range(visible_matrix.shape[1]):
        unvisible_pts = np.argwhere(pts_bone_visibility[:, c] == 0).squeeze(1)
        visible_pts = np.argwhere(pts_bone_visibility[:, c] == 1).squeeze(1)
        if len(visible_pts) == 0:
            visible_matrix[:, c] = pts_bone_dist[:, c]
            continue
        for r in unvisible_pts:
            dist1 = np.min(surface_geodesic[r, visible_pts])
            nn_visible = visible_pts[np.argmin(surface_geodesic[r, visible_pts])]
            if np.isinf(dist1):
                visible_matrix[r, c] = 8.0 + pts_bone_dist[r, c]
            else:
                visible_matrix[r, c] = dist1 + visible_matrix[nn_visible, c]
    if subsampling:
        nn_dist = np.sum((mesh_v[:, np.newaxis, :] - subsamples[np.newaxis, ...])**2, axis=2)
        nn_ind = np.argmin(nn_dist, axis=1)
        visible_matrix = visible_matrix[nn_ind, :]
        os.remove(mesh_filename.replace(".obj", "_simplified.obj"))
    return visible_matrix


def predict_skinning(input_data, pred_skel, skin_pred_net, surface_geodesic, mesh_filename, subsampling=False):
    """
    predict skinning
    :param input_data: wrapped input data
    :param pred_skel: predicted skeleton
    :param skin_pred_net: network to predict skinning weights
    :param surface_geodesic: geodesic distance matrix of all vertices
    :param mesh_filename: mesh filename
    :return: predicted rig with skinning weights information
    """
    global device, output_folder
    num_nearest_bone = 5
    bones, bone_names, bone_isleaf = get_bones(pred_skel)
    mesh_v = input_data.pos.data.cpu().numpy()
    print("     calculating volumetric geodesic distance from vertices to bone. This step takes some time...")
    geo_dist = calc_geodesic_matrix(bones, mesh_v, surface_geodesic, mesh_filename, subsampling=subsampling)
    input_samples = []  # joint_pos (x, y, z), (bone_id, 1/D)*5
    loss_mask = []
    skin_nn = []
    for v_id in range(len(mesh_v)):
        geo_dist_v = geo_dist[v_id]
        bone_id_near_to_far = np.argsort(geo_dist_v)
        this_sample = []
        this_nn = []
        this_mask = []
        for i in range(num_nearest_bone):
            if i >= len(bones):
                this_sample += bones[bone_id_near_to_far[0]].tolist()
                this_sample.append(1.0 / (geo_dist_v[bone_id_near_to_far[0]] + 1e-10))
                this_sample.append(bone_isleaf[bone_id_near_to_far[0]])
                this_nn.append(0)
                this_mask.append(0)
            else:
                skel_bone_id = bone_id_near_to_far[i]
                this_sample += bones[skel_bone_id].tolist()
                this_sample.append(1.0 / (geo_dist_v[skel_bone_id] + 1e-10))
                this_sample.append(bone_isleaf[skel_bone_id])
                this_nn.append(skel_bone_id)
                this_mask.append(1)
        input_samples.append(np.array(this_sample)[np.newaxis, :])
        skin_nn.append(np.array(this_nn)[np.newaxis, :])
        loss_mask.append(np.array(this_mask)[np.newaxis, :])

    skin_input = np.concatenate(input_samples, axis=0)
    loss_mask = np.concatenate(loss_mask, axis=0)
    skin_nn = np.concatenate(skin_nn, axis=0)
    skin_input = torch.from_numpy(skin_input).float()
    input_data.skin_input = skin_input
    input_data.to(device)

    skin_pred = skin_pred_net(input_data)
    skin_pred = torch.softmax(skin_pred, dim=1)
    skin_pred = skin_pred.data.cpu().numpy()
    skin_pred = skin_pred * loss_mask

    skin_nn = skin_nn[:, 0:num_nearest_bone]
    skin_pred_full = np.zeros((len(skin_pred), len(bone_names)))
    for v in range(len(skin_pred)):
        for nn_id in range(len(skin_nn[v, :])):
            skin_pred_full[v, skin_nn[v, nn_id]] = skin_pred[v, nn_id]
    print("     filtering skinning prediction")
    tpl_e = input_data.tpl_edge_index.data.cpu().numpy()
    skin_pred_full = post_filter(skin_pred_full, tpl_e, num_ring=1)
    skin_pred_full[skin_pred_full < np.max(skin_pred_full, axis=1, keepdims=True) * 0.35] = 0.0
    skin_pred_full = skin_pred_full / (skin_pred_full.sum(axis=1, keepdims=True) + 1e-10)
    skel_res = assemble_skel_skin(pred_skel, skin_pred_full)
    return skel_res


def tranfer_to_ori_mesh(filename_ori, filename_remesh, pred_rig):
    """
    convert the predicted rig of remeshed model to the rig of the original model.
    Just assign skinning weight based on nearest neighbor
    :param filename_ori: original mesh filename
    :param filename_remesh: remeshed mesh filename
    :param pred_rig: predicted rig
    :return: predicted rig for original mesh
    """
    mesh_remesh = o3d.io.read_triangle_mesh(filename_remesh)
    mesh_ori = o3d.io.read_triangle_mesh(filename_ori)
    tranfer_rig = Info()

    vert_remesh = np.asarray(mesh_remesh.vertices)
    vert_ori = np.asarray(mesh_ori.vertices)

    vertice_distance = np.sqrt(np.sum((vert_ori[np.newaxis, ...] - vert_remesh[:, np.newaxis, :]) ** 2, axis=2))
    vertice_raw_id = np.argmin(vertice_distance, axis=0)  # nearest vertex id on the fixed mesh for each vertex on the remeshed mesh

    tranfer_rig.root = pred_rig.root
    tranfer_rig.joint_pos = pred_rig.joint_pos
    new_skin = []
    for v in range(len(vert_ori)):
        skin_v = [v]
        v_nn = vertice_raw_id[v]
        skin_v += pred_rig.joint_skin[v_nn][1:]
        new_skin.append(skin_v)
    tranfer_rig.joint_skin = new_skin
    return tranfer_rig


if __name__ == '__main__':
    input_folder = "data/"

    # downsample_skinning is used to speed up the calculation of volumetric geodesic distance
    # and to save cpu memory in skinning calculation.
    # Change to False to be more accurate but less efficient.
    downsample_skinning = True

    # load all weights
    print("loading all networks...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    skinNet = skinnet(nearest_bone=5, use_Dg=False, use_Lf=False)
    skinNet_checkpoint = torch.load('checkpoints/skinnet/model_best.pth.tar')
    skinNet.load_state_dict(skinNet_checkpoint['state_dict'])
    skinNet.to(device)
    skinNet.eval()
    print("     skinning prediction network loaded.")

    # Here we provide 16~17 examples. For best results, we will need to override the learned bandwidth and its associated threshold
    # To process other input characters, please first try the learned bandwidth (0.0429 in the provided model), and the default threshold 1e-5.
    # We also use these two default parameters for processing all test models in batch.

    #model_id, bandwidth, threshold = "smith", None, 1e-5
    #model_id, bandwidth, threshold = "17872", 0.045, 0.75e-5
    #model_id, bandwidth, threshold = "8210", 0.05, 1e-5
    #model_id, bandwidth, threshold = "8330", 0.05, 0.8e-5
    model_id, bandwidth, threshold = "9477", 0.043, 2.5e-5
    #model_id, bandwidth, threshold = "17364", 0.058, 0.3e-5
    #model_id, bandwidth, threshold = "15930", 0.055, 0.4e-5
    #model_id, bandwidth, threshold = "8333", 0.04, 2e-5
    #model_id, bandwidth, threshold = "8338", 0.052, 0.9e-5
    #model_id, bandwidth, threshold = "3318", 0.03, 0.92e-5
    #model_id, bandwidth, threshold = "15446", 0.032, 0.58e-5
    #model_id, bandwidth, threshold = "1347", 0.062, 3e-5
    #model_id, bandwidth, threshold = "11814", 0.06, 0.6e-5
    #model_id, bandwidth, threshold = "2982", 0.045, 0.3e-5
    #model_id, bandwidth, threshold = "2586", 0.05, 0.6e-5
    #model_id, bandwidth, threshold = "8184", 0.05, 0.4e-5
    #model_id, bandwidth, threshold = "9000", 0.035, 0.16e-5

    # create data used for inferece
    print("creating data for model ID {:s}".format(model_id))
    mesh_filename = os.path.join(input_folder, 'obj_remesh/{:s}.obj'.format(model_id))
    skel_filename = os.path.join(input_folder, 'rig_info/{:s}.txt'.format(model_id))

    data, surface_geodesic, translation_normalize, scale_normalize = create_single_data(mesh_filename)
    data.to(device)

    skeleton = Info(skel_filename)
    print("predicting skinning")
    pred_rig = predict_skinning(data, skeleton, skinNet, surface_geodesic,
                                mesh_filename.replace("_remesh.obj", "_normalized.obj"),
                                subsampling=downsample_skinning)

    # here we reverse the normalization to the original scale and position
    pred_rig.normalize(scale_normalize, -translation_normalize)

    print("Saving result")
    if True:
        # here we use original mesh tesselation (without remeshing)
        mesh_filename_ori = os.path.join(input_folder, '{:s}_ori.obj'.format(model_id))
        pred_rig = tranfer_to_ori_mesh(mesh_filename_ori, mesh_filename, pred_rig)
        pred_rig.save(mesh_filename_ori.replace('.obj', '_rig.txt'))
    else:
        # here we use remeshed mesh
        pred_rig.save(mesh_filename.replace('.obj', '_rig.txt'))
    print("Done!")
