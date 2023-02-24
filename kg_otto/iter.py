import pandas as pd
from collections import Counter
from operator import itemgetter

import tqdm

from kg_otto.config import SESSION_COL, ITEM_COL, SCORE_COL


def iter_row_values(df, cols=None):
    if isinstance(df, pd.DataFrame):
        cols = cols or list(df.columns)
        values = [df[col] for col in cols]
        yield from zip(*values)
    elif isinstance(df, dict):
        cols = cols or list(df[0].keys())
        yield from map(itemgetter(*cols), df)


def gen_grp(df, grp_col, val_col):
    # noinspection PyTypeChecker
    diff: pd.Series = df[grp_col] != df[grp_col].shift(1)
    index = diff[diff].index.tolist()
    index.append(len(df))
    val_idx = df.columns.tolist().index(val_col)
    vv = df.values[:, val_idx]

    for i, j in zip(index, index[1:]):
        yield df[grp_col].iloc[i], vv[i:j]


def gen_grp_cols(df, grp_cols, val_cols, sort=True):
    if sort:
        df = df.sort_values(grp_cols, ignore_index=True)
    diff = (df[grp_cols] != df[grp_cols].shift(1)).any(axis=1)
    index = diff[diff].index.tolist()
    index.append(len(df))

    val_idx = [df.columns.tolist().index(val_col) for val_col in val_cols]
    vv = df.values[:, val_idx]
    grp_idx = [df.columns.tolist().index(grp_col) for grp_col in grp_cols]
    gg = df.values[:, grp_idx]

    for i, j in zip(index, index[1:]):
        yield gg[i], vv[i:j]


def make_counts(df):
    ww = list(gen_grp(df, SESSION_COL, ITEM_COL))
    df = pd.DataFrame(ww, columns=[SESSION_COL, ITEM_COL])
    df[ITEM_COL] = df[ITEM_COL].apply(lambda x: list(Counter(x.tolist()).items()))
    df = df.explode(ITEM_COL, ignore_index=True)
    df[SCORE_COL] = df[ITEM_COL].apply(itemgetter(1))
    df[ITEM_COL] = df[ITEM_COL].apply(itemgetter(0))
    return df


def iter_tqdm(iter_res, **kwargs):
    results = []
    with tqdm.tqdm(**kwargs) as timer:
        for res in iter_res:
            results.append(res)
            timer.update()
    return results
