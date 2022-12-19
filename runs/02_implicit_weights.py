import logging
from read_data import get_data, do_eval
from kg_otto.utils import coo_to_pd
from kg_otto.imp.imp import ImplicitPred, wgt2cls


train_df, test_df, test_labels = get_data()

train_sparse = None
encoder = None
k_neighbours = 100


for cf_type in wgt2cls.keys():
    for agg in ["max", "sum"]:
        name = f"implicit_sim_{cf_type}_{agg}_{k_neighbours}"
        logging.info(f"Start {name} run")

        imp = ImplicitPred(k_neighbours, cf_type, agg)

        if train_sparse is None:
            imp.fit(train_df)
            train_sparse = imp.coo.tocsr()
            encoder = imp.encoder
        else:
            imp.encoder = encoder
            imp.fit(train_sparse, from_sparse=True)

        logging.info(f"Save {name} similarity")
        sim = imp.model.similarity.tocoo()
        coo_to_pd(sim).to_parquet(name + '.parquet')

        pred = imp.predict(test_df, parallel=6)
        pred.to_csv(f"predict_{name}.csv")

        print(name)
        print(do_eval(pred, test_labels))
        print()

