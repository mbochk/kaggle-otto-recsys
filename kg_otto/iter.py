import pandas as pd
from collections import Counter
from operator import itemgetter

from kg_otto.config import SESSION_COL, ITEM_COL


def iter_row_values(df, cols=None):
    cols = cols or list(df.columns)
    values = [df[col] for col in cols]
    yield from zip(*values)


def gen_grp(df, grp_col, val_col):
    diff = df[grp_col] != df[grp_col].shift(1)
    index = diff[diff].index.tolist()
    index.append(len(df))
    val_idx = df.columns.tolist().index(val_col)
    vv = df.values[:, val_idx]

    for i, j in zip(index, index[1:]):
        yield df[grp_col].iloc[i], vv[i:j]


def make_counts(df):
    ww = list(gen_grp(df.iloc[:M], SESSION_COL, ITEM_COL))
    df = pd.DataFrame(ww, columns=[SESSION_COL, ITEM_COL])
    df[ITEM_COL] = df[ITEM_COL].apply(lambda x: list(Counter(x.tolist()).items()))
    df = df.explode(ITEM_COL, ignore_index=True)
    df[SCORE_COL] = df[ITEM_COL].apply(itemgetter(1))
    df[ITEM_COL] = df[ITEM_COL].apply(itemgetter(0))
    return df