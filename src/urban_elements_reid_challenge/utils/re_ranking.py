"""
k-Reciprocal Re-ranking for Person/Object Re-ID.

Reference:
    Zhong Z, Zheng L, Cao D, et al. "Re-ranking Person Re-identification with
    k-reciprocal Encoding." CVPR 2017.
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf

Two APIs are provided:
    re_ranking(q_g_dist, q_q_dist, g_g_dist, ...)
        Distance-matrix API. Inputs are pre-computed cosine distance matrices.
        This is the primary API used throughout evaluation scripts.

    re_ranking_from_features(qf, gf, ...)
        Feature-vector API. Inputs are raw L2-normalised feature tensors.
        Computes Euclidean distances internally (compatible with TransReID/PAT
        training loop metrics).
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Primary API — distance matrix inputs (cosine distances)
# ---------------------------------------------------------------------------

def _k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """k-Reciprocal re-ranking from pre-computed cosine distance matrices.

    Args:
        q_g_dist  (np.ndarray): shape [nQ, nG], query-gallery cosine distances.
        q_q_dist  (np.ndarray): shape [nQ, nQ], query-query cosine distances.
        g_g_dist  (np.ndarray): shape [nG, nG], gallery-gallery cosine distances.
        k1        (int): neighbourhood size for k-reciprocal set (default 20).
        k2        (int): neighbourhood size for query expansion (default 6).
        lambda_value (float): balancing weight between Jaccard and original dist
                              (default 0.3).

    Returns:
        final_dist (np.ndarray): re-ranked distance matrix, shape [nQ, nG].
    """
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    # Convert cosine similarity to squared Euclidean-like distance
    original_dist = 2. - 2 * original_dist
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    V = np.zeros_like(original_dist, dtype=np.float32)
    # Use argpartition (top-k) for speed
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))

    for i in range(all_num):
        k_reciprocal_index = _k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = _k_reciprocal_neigh(
                initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,
                                  k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(all_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    return final_dist[:query_num, query_num:]


# ---------------------------------------------------------------------------
# Secondary API — raw feature-vector inputs (Euclidean distance)
# ---------------------------------------------------------------------------

def re_ranking_from_features(qf, gf, k1=20, k2=6, lambda_value=0.3,
                              local_distmat=None, only_local=False):
    """k-Reciprocal re-ranking from raw feature vectors.

    Computes pairwise Euclidean distances internally.  Compatible with the
    TransReID/PAT training-loop metrics where features are extracted as
    torch tensors.

    Args:
        qf            (torch.Tensor): query features,   shape [nQ, d].
        gf            (torch.Tensor): gallery features, shape [nG, d].
        k1, k2, lambda_value: same as ``re_ranking``.
        local_distmat (np.ndarray | None): optional local distance matrix to add.
        only_local    (bool): use only local_distmat, skip feature distances.

    Returns:
        final_dist (np.ndarray): re-ranked distance matrix, shape [nQ, nG].
    """
    query_num = qf.size(0)
    all_num = query_num + gf.size(0)

    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([qf, gf])
        distmat = (torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
                   + torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t())
        distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist, dtype=np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            cand_forward = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            cand_backward = initial_rank[cand_forward, :int(np.around(k1 / 2)) + 1]
            fi_cand = np.where(cand_backward == candidate)[0]
            cand_k_recip = cand_forward[fi_cand]
            if len(np.intersect1d(cand_k_recip, k_reciprocal_index)) > 2 / 3 * len(cand_k_recip):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, cand_k_recip)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = [np.where(V[:, i] != 0)[0] for i in range(gallery_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist, V, jaccard_dist
    return final_dist[:query_num, query_num:]
