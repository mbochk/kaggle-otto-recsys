import logging
import numpy as np
import pandas as pd

from multiprocessing import Pool
from scipy.sparse import coo_matrix
from implicit.nearest_neighbours import ItemItemRecommender, CosineRecommender, TFIDFRecommender, BM25Recommender
from sklearn.preprocessing import OrdinalEncoder

from kg_otto.config import SESSION_COL, ITEM_COL, SCORE_COL

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

    def predict(self, x):
        x = x.copy()
        logging.info("Transform data for prediction")
        ij = self.encoder.transform(x[[SESSION_COL, ITEM_COL]])
        x['item_index'] = ij[:, 1]
        x = x[x.item_index != self.UNK_VALUE]
        x = x.groupby(SESSION_COL).item_index.apply(set)
        x.method = self.u2i_agg

        data_rows = ((val, self.u2i_agg, self.top_k) for val in x.values)

        global sim
        sim = self.model.similarity

        with Pool(processes=self.PARALLEL) as pool:
            logging.info(f"Apply aggregation in multiprocessing")
            predict = pool.map(predict_apply, data_rows, chunksize=10000)
        del sim

        x = pd.DataFrame(x)
        x["predict"] = predict
        x = x.explode("predict")
        from operator import itemgetter
        x[ITEM_COL] = x["predict"].apply(itemgetter(0)).astype(int)
        x[SCORE_COL] = x["predict"].apply(itemgetter(1))
        x = x[[ITEM_COL, SCORE_COL]].reset_index()
        return x


def max_aggr(rows, sim, top_k):
    v = sim[list(rows)].max(axis=0)
    index = np.argsort(v.data)[:-1-top_k:-1]
    return v.col[index], v.data[index]


def sum_aggr(rows, sim, top_k):
    v = sum(sim[list(rows)])
    index = np.argsort(v.data)[:-1-top_k:-1]
    return v.indices[index], v.data[index]


def predict_apply(data):
    global sim
    rows, u2i_agg, top_k = data
    if u2i_agg == "max":
        return max_aggr(rows, sim, top_k)
    else:
        return sum_aggr(rows, sim, top_k)
