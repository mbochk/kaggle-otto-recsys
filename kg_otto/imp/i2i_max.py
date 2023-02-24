import os
import logging
import pandas as pd

from collections import defaultdict, Counter
from tqdm import tqdm
from operator import itemgetter

from kg_otto.data import get_test
from kg_otto import set_log_level
from kg_otto.iter import gen_grp, iter_row_values
from kg_otto.partitioned import PartitionedDataFrame


def all_i2i_count(df):
    lww = list(gen_grp(df, 'session', 'aid'))
    best_i2i = defaultdict(Counter)

    for grp in lww:
        items = set(map(int, grp[1]))
        for i in items:
            best_i2i[i].update(items)
    i2i_df_data = [
        (aid, aid2, score)
        for aid, aid_ngb in best_i2i.items()
        for aid2, score in aid_ngb.items()
    ]
    i2i_df = pd.DataFrame(i2i_df_data, columns=['aid', 'aid2', 'score'])
    i2i_df.sort_values(['aid', 'aid2'], inplace=True)
    return i2i_df


def collect_total_predict(train, predict):
    p_train = PartitionedDataFrame(train)
    if predict in os.listdir():
        output = PartitionedDataFrame(predict)
    else:
        output = PartitionedDataFrame(predict)
        output.create()
        p_train.mp_apply(all_i2i_count, output)
    return output


def collect_i2i_v2(part_predict):
    from operator import itemgetter
    from kg_otto.data import get_test
    test_labels = get_test(merge_test=True)

    truth_counts = test_labels.ground_truth.apply(lambda x: Counter()).tolist()
    data_iter = list(iter_row_values(test_labels, cols=['ground_truth', 'data']))

    p_pred = PartitionedDataFrame(part_predict)

    for pt in tqdm(p_pred.partitions):
        logging.info(f"Read {pt}")
        df = p_pred.get_df(pt)

        logging.info("Make i2i")
        i2i = defaultdict(Counter)
        for aid, aid2, score in iter_row_values(df):
            i2i[aid][aid2] += score

        logging.info("Start summing")
        for (truth, data), tcount in zip(data_iter, truth_counts):
            op = itemgetter(*data)
            tr_count = {int(tr): op(i2i[tr]) for tr in truth}
            tr_count = {tr: int(sum(val)) if isinstance(val, tuple) else int(val)
                        for tr, val in tr_count.items()}
            tcount.update(tr_count)
    return test_labels, truth_counts


def all_i2i_val(df):
    test_labels = get_test(merge_test=True)

    i2i = defaultdict(Counter)
    for aid, aid2, score in iter_row_values(df):
        i2i[aid][aid2] += score

    truth_counts = test_labels.ground_truth.apply(lambda x: Counter()).tolist()
    data_iter = iter_row_values(test_labels, cols=['ground_truth', 'data'])

    for (truth, data), tcount in zip(data_iter, truth_counts):
        op = itemgetter(*data)
        tr_count = {int(tr): op(i2i[tr]) for tr in truth}
        tr_count = {tr: int(sum(val)) if isinstance(val, tuple) else int(val)
                    for tr, val in tr_count.items()}
        tcount.update(tr_count)

    test_labels['aid'] = truth_counts
    val = test_labels[['session', 'type', 'aid']].explode('aid')
    val['score'] = test_labels[['aid']].applymap(lambda x: x.values()).explode("aid").values
    return val


def collect_i2i_eval(predict, val):
    p_pred = PartitionedDataFrame(predict)
    p_val = PartitionedDataFrame(val)
    p_val.create()
    p_pred.mp_apply(all_i2i_val, p_val)


def main():
    train = "partitioned_train.parquet"
    predict = 'partitioned_predict_i2i_full.parquet'
    val = 'partitioned_eval_i2i_full.parquet'
    set_log_level(20)

    logging.info("Predict on partitioned")
    collect_total_predict(train, predict)

    logging.info("Evaluate predict on test")
    collect_i2i_eval(predict, val)


if __name__ == "__main__":
    main()
