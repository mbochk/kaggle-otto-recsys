import logging
import numpy as np
import pandas as pd

from multiprocessing import Pool
from scipy.sparse import coo_matrix
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender
from sklearn.preprocessing import OrdinalEncoder

from kg_otto.config import SESSION_COL, ITEM_COL, SCORE_COL

from kg_otto.utils import add_types
from kg_otto.iter import iter_tqdm

wgt2cls = {
    "item": ItemItemRecommender,
    "cos": CosineRecommender,
    "tf-idf": TFIDFRecommender,
    "bm25": BM25Recommender
}


class ImplicitPred:
    UNK_VALUE = -1
    PARALLEL = 20

    def __init__(self, K, cf_type: str, u2i_agg: str):
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=self.UNK_VALUE, dtype='int')
        self.model: ItemItemRecommender = wgt2cls[cf_type](K, num_threads=self.PARALLEL)
        self.u2i_agg = u2i_agg
        self.top_k = K

    def fit(self, x, from_sparse=False):
        logging.info("Start ImplicitPred model fit")
        if not from_sparse:
            x = x.groupby([SESSION_COL, ITEM_COL], as_index=False).ts.agg('count')
            ij = self.encoder.fit_transform(x[[SESSION_COL, ITEM_COL]])
            c = coo_matrix((x.ts.astype('float'), list(ij.T)))
            self.coo = c
        else:
            c = x
        logging.info("Start implicit model fit")
        self.model.fit(c)

    def _parallel_predict(self, data_rows, parallel):
        global sim
        sim = self.model.similarity
        if isinstance(parallel, bool):
            parallel = self.PARALLEL

        with Pool(processes=parallel) as pool:
            logging.info(f"Apply aggregation in multiprocessing")
            iter_res = pool.imap(predict_apply, data_rows, chunksize=1000)
            res = iter_tqdm(iter_res, total=len(data_rows), smoothing=0)
        del sim
        return res

    def predict(self, x, parallel=True, types=None):
        x = x.copy()
        logging.info("Transform data for prediction")
        ij = self.encoder.transform(x[[SESSION_COL, ITEM_COL]])
        x['item_index'] = ij[:, 1]
        x = x[x.item_index != self.UNK_VALUE]
        x = x.groupby(SESSION_COL).item_index.apply(set)

        data_rows = [(val, self.u2i_agg, self.top_k) for val in x.values]
        if parallel:
            predict = self._parallel_predict(data_rows, parallel=parallel)
        else:
            predict = [predict_apply(row, self.model.similarity) for row in data_rows]

        x = pd.DataFrame(x)
        x["predict"] = predict
        x = x.explode("predict")
        items = x.iloc[0::2].explode("predict").reset_index()
        items.rename(columns=dict(predict=ITEM_COL), inplace=True)
        items[ITEM_COL] = items[ITEM_COL].astype(int)
        items[SCORE_COL] = x.iloc[1::2].explode("predict")["predict"].values
        items = items[[SESSION_COL, ITEM_COL, SCORE_COL]]
        return add_types(items)


def max_aggr(rows, sim, top_k):
    v = sim[list(rows)].max(axis=0)
    index = np.argsort(v.data)[:-1-top_k:-1]
    return v.col[index], v.data[index]


def sum_aggr(rows, sim, top_k):
    v = sum(sim[list(rows)])
    index = np.argsort(v.data)[:-1-top_k:-1]
    return v.indices[index], v.data[index]


def predict_apply(data, ssim=None):
    if ssim is None:
        global sim
    else:
        sim = ssim
    rows, u2i_agg, top_k = data
    if u2i_agg == "max":
        return max_aggr(rows, sim, top_k)
    else:
        return sum_aggr(rows, sim, top_k)
