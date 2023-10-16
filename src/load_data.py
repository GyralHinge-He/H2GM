import scipy.io as io
import torch
import numpy as np
import os
from torch.nn.parameter import Parameter
import  nibabel as nib
import matplotlib.pyplot as plt


device = torch.device("cuda:0")

def load_data_aparc(filename1, filename2, GT_path):
    fixed_graph = io.loadmat(filename1)
    fixed_fname = filename1[-31:-25]
    warp_graph = io.loadmat(filename2)
    warp_fname = filename2[-31:-25]

    Ground_truth = io.loadmat(GT_path + fixed_fname + '_2_' + warp_fname + '.L.GT.matrix.mat')
    Ground_truth = Ground_truth['GroundTruth_matrix']
    Ground_truth = torch.DoubleTensor(Ground_truth)

    fixed_delete_index = torch.sum(Ground_truth, dim=1)
    warp_delete_index = torch.sum(Ground_truth, dim=0)

    fixed_delete_index1 = torch.where(fixed_delete_index == 1)

    warp_delete_index1 = torch.where(warp_delete_index == 1)

    Ground_truth = Ground_truth[fixed_delete_index1[0].tolist(), :]
    Ground_truth = Ground_truth[:, warp_delete_index1[0].tolist()]

    fixed_graph_node_feat = fixed_graph['feat_node']
    fixed_graph_node_feat = fixed_graph_node_feat[:, 0:174]
    fixed_graph_node_feat = fixed_graph_node_feat[fixed_delete_index1[0].tolist(), :]

    fixed_hypergraph_matrix = np.zeros((fixed_graph_node_feat.shape[0], 5 + 35 + fixed_graph_node_feat.shape[0]))

    for t in range(0, fixed_graph_node_feat.shape[0]):
        fixed_hypergraph_matrix[t, int(fixed_graph_node_feat[t, 1]) - 1] = fixed_hypergraph_matrix[t, int(
            fixed_graph_node_feat[t, 1]) - 1] + 1

    for t in range(0, fixed_graph_node_feat.shape[0]):
        fixed_hypergraph_matrix[t, int(fixed_graph_node_feat[t, 2]) + 4] = fixed_hypergraph_matrix[t, int(
            fixed_graph_node_feat[t, 2]) + 4] + 1

    for t in range(0, fixed_graph_node_feat.shape[0]):
        fixed_hypergraph_matrix[t, 5 + 35 + t] = fixed_hypergraph_matrix[t, 5 + 35 + t] + 1

    fixed_hypergraph_matrix = torch.DoubleTensor(fixed_hypergraph_matrix)
    fixed_delete_index1 = torch.where(torch.sum(fixed_hypergraph_matrix, dim=0) != 0)
    fixed_hypergraph_matrix = fixed_hypergraph_matrix[:, fixed_delete_index1[0].tolist()]

    fixed_brainarea_matrix = fixed_graph_node_feat[:, 1:3]

    fixed_graph_node_feat = fixed_graph_node_feat[:, 3:174]

    warp_graph_node_feat = warp_graph['feat_node']
    warp_graph_node_feat = warp_graph_node_feat[:, 0:174]
    warp_graph_node_feat = warp_graph_node_feat[warp_delete_index1[0].tolist(), :]

    warp_hypergraph_matrix = np.zeros((warp_graph_node_feat.shape[0], 5 + 35 + warp_graph_node_feat.shape[0]))

    for t in range(0, warp_graph_node_feat.shape[0]):
        warp_hypergraph_matrix[t, int(warp_graph_node_feat[t, 1]) - 1] = warp_hypergraph_matrix[
                                                                             t, int(warp_graph_node_feat[t, 1]) - 1] + 1

    for t in range(0, warp_graph_node_feat.shape[0]):
        warp_hypergraph_matrix[t, int(warp_graph_node_feat[t, 2]) + 4] = warp_hypergraph_matrix[
                                                                             t, int(warp_graph_node_feat[t, 2]) + 4] + 1

    for t in range(0, warp_graph_node_feat.shape[0]):
        warp_hypergraph_matrix[t, 5 + 35 + t] = warp_hypergraph_matrix[t, 5 + 35 + t] + 1

    warp_hypergraph_matrix = torch.DoubleTensor(warp_hypergraph_matrix)
    warp_delete_index1 = torch.where(torch.sum(warp_hypergraph_matrix, dim=0) != 0)
    warp_hypergraph_matrix = warp_hypergraph_matrix[:, warp_delete_index1[0].tolist()]

    warp_brainarea_matrix = warp_graph_node_feat[:, 1:3]

    warp_graph_node_feat = warp_graph_node_feat[:, 3:174]

    fixed_graph_node_feat = torch.DoubleTensor(fixed_graph_node_feat)
    warp_graph_node_feat = torch.DoubleTensor(warp_graph_node_feat)



    gt = torch.zeros(fixed_hypergraph_matrix.shape[1], warp_hypergraph_matrix.shape[1])
    fixed_shape = (fixed_hypergraph_matrix.shape[1] - fixed_hypergraph_matrix.shape[0])
    warp_shape = (warp_hypergraph_matrix.shape[1] - warp_hypergraph_matrix.shape[0])
    gt[0:fixed_shape, 0:warp_shape] = gt[0:fixed_shape, 0:warp_shape] + torch.diag(torch.ones(fixed_shape))
    gt[fixed_shape: fixed_hypergraph_matrix.shape[1], warp_shape: warp_hypergraph_matrix.shape[1]] = gt[fixed_shape:
                                                                                                        fixed_hypergraph_matrix.shape[
                                                                                                            1],
                                                                                                     warp_shape:
                                                                                                     warp_hypergraph_matrix.shape[
                                                                                                         1]] + Ground_truth
    Ground_truth = gt

    # plt.imshow(fixed_graph_node_feat.detach().cpu().numpy())
    fixed_graph_node_feat = fixed_graph_node_feat.to(device)
    warp_graph_node_feat = warp_graph_node_feat.to(device)
    fixed_hypergraph_matrix = fixed_hypergraph_matrix.to(device)
    warp_hypergraph_matrix = warp_hypergraph_matrix.to(device)
    # fixed_hypergraph_matrix_matmul = fixed_hypergraph_matrix_matmul.to(device)
    # warp_hypergraph_matrix_matmul = warp_hypergraph_matrix_matmul.to(device)
    Ground_truth = Ground_truth.to(device)

    return {
        'fixed_graph_node_feat': fixed_graph_node_feat,
        'warp_graph_node_feat': warp_graph_node_feat,
        'fixed_hypergraph_matrix': fixed_hypergraph_matrix,
        'warp_hypergraph_matrix': warp_hypergraph_matrix,
        'fixed_brainarea_matrix': fixed_brainarea_matrix,
        'warp_brainarea_matrix': warp_brainarea_matrix,
        'Ground_truth': Ground_truth,
    }
