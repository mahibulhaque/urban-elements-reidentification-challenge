"""Camera-Aware Jaccard (CAJ) re-ranking.

CKRNNS = camera-aware K-Reciprocal Nearest Neighbors (split intra/inter-camera)
CLQE   = camera-aware Local Query Expansion
"""
import numpy as np
import torch


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking_caj(probFea, galFea, cids, k1, k2, lambda_value,
                   ckrnns, k1_intra, k1_inter, clqe, k2_intra, k2_inter,
                   local_distmat=None, only_local=False, q_g_distmat=None):
    query_num = probFea.size(0)
    gallery_num = query_num + galFea.size(0)
    all_num = gallery_num
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea, galFea])
        distmat = (torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
                   + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t())
        distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat

    if q_g_distmat is not None:
        original_dist[:query_num, query_num:] = q_g_distmat
        original_dist[query_num:, :query_num] = q_g_distmat.t()

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    cam_mask = (cids.reshape(-1, 1) == cids.reshape(1, -1)).numpy()

    inter_rank = np.argpartition(original_dist + 999.0 * cam_mask, range(k1_inter + 2))
    nn_inter = [k_reciprocal_neigh(inter_rank, i, k1_inter) for i in range(all_num)]
    intra_rank = np.argpartition(original_dist + 999.0 * (~cam_mask), range(k1_intra + 2))
    nn_intra = [k_reciprocal_neigh(intra_rank, i, k1_intra) for i in range(all_num)]

    for i in range(all_num):
        if ckrnns:
            k_reciprocal_index = np.append(nn_intra[i], nn_inter[i])
            k_reciprocal_expansion_index = k_reciprocal_index
        else:
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2.)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2.)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,
                                                             candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]

    V_qe = np.zeros_like(V, dtype=np.float32)
    for i in range(all_num):
        if clqe:
            k2nn = np.append(intra_rank[i, :k2_intra], inter_rank[i, :k2_inter])
        else:
            k2nn = initial_rank[i, :k2]
        V_qe[i, :] = np.mean(V[k2nn, :], axis=0)
    V = V_qe
    del V_qe

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
