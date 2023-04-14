import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.metrics import ndcg_score


def eval_score(score, target, N_groups):
    target = target.reshape([N_groups, -1])
    score = score.reshape([N_groups, -1])
    return ndcg_score(target, score)


def eval_w(w, ff, target):
    score = ff.dot(w)
    return eval_score(score, target)


def calc_pairs(w, ff, target, grp, S):
    score = ff.dot(w)
    rk = pd.Series(score).groupby(grp).rank("first", ascending=False)

    data = dict(
        target=pd.Series(target),
        grp=pd.Series(grp),
        score=pd.Series(score),
        rk=rk,
        idx=pd.Series(np.arange(len(score)))
    )
    aux_df = pd.concat(data, axis=1)

    pairs = pd.merge(aux_df[aux_df.target != 0], aux_df, on='grp', suffixes=('', '_x'))
    pairs = pairs[(pairs.rk != pairs.rk_x) & (pairs.target > pairs.target_x)]

    s_diff = S * (pairs.score - pairs.score_x)
    pairs['cost'] = - np.log(expit(s_diff))
    pairs['L_ij'] = - S * expit(-s_diff)
    pairs['L_ij_ndcg'] = pairs['L_ij'] * np.abs((pairs.target - pairs.target_x) * (1. / pairs.rk - 1. / pairs.rk_x))
    return pairs


def calc_L_i(pairs, key="L_ij"):
    # combine L_i from L_ij
    L_i = pairs.groupby(['grp', 'idx'])[key].sum()
    L_j = pairs.groupby(['grp', 'idx_x'])[key].sum()

    L_index = set(L_i.index)
    L_index.update(L_j.index)

    L_i = L_i.reindex(L_index, fill_value=0.) - L_j.reindex(L_index, fill_value=0.)
    L_i = L_i.sort_index().values
    return L_i


def update_lr(w, ff, target, grp, S, lr=0.01):
    pairs = calc_pairs(w, ff, target, grp, S)
    L_i = calc_L_i(pairs, key="L_ij_ndcg")
    dC_dw = (ff * calc_L_i(pairs, key="L_ij_ndcg")[:, None]).mean(axis=0)
    w -= dC_dw * lr
