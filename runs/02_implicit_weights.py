from kg_otto import *
from kg_otto.utils import read_test_labels, coo_to_pd
from kg_otto.imp.imp import ImplicitPred, wgt2cls
from pathlib import Path

data_path = Path("/home/mikhail.bochkarev/study/")

train_df = pd.read_parquet(data_path/"split_train.parquet")

test_df = pd.read_parquet(data_path/"test.parquet")
test_labels = read_test_labels(data_path/"test_labels.parquet")


train_sparse = None
encoder = None
k_neighbours = 100


for cf_type in wgt2cls.keys():
    for agg in ["max", "sum"]:
        name = f"implicit_sim_{cf_type}_{agg}_{k_neighbours}"

        imp = ImplicitPred(k_neighbours, cf_type, agg)

        if train_sparse is None:
            imp.fit(train_df)
            train_sparse = imp.coo.to_csr()
            encoder = imp.encoder
        else:
            imp.encoder = encoder
            imp.fit(train_sparse, from_sparse=True)

        sim = imp.model.similarity.tocoo()
        coo_to_pd(sim).to_csv(name)

        pred = imp.predict(test_df)

        print(name)
        print(do_eval(pred, test_labels))
        print()

