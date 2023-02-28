import logging
import pandas as pd

from collections import defaultdict, Counter
from operator import itemgetter

from kg_otto.data import get_test
from kg_otto import set_log_level
from kg_otto.iter import gen_grp, iter_row_values
from kg_otto.partitioned import PartitionedDataFrame
from kg_otto.config import TRUTH_COL


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
    p_train, p_predict = PartitionedDataFrame(train), PartitionedDataFrame(predict)
    p_predict_tmp = PartitionedDataFrame(predict + '_tmp')

    def merge(df):
        df = df.groupby(["aid", "aid2"], as_index=False).sum()
        df.sort_values(["aid", "score"], ascending=[True, False], inplace=True)
        df['rank'] = df.groupby("aid").score.rank(method='first', ascending=False).astype('int')
        return df

    def repartition(df):
        df['aid_pt'] = df.aid // 18000
        return df

    p_train.mp_apply(all_i2i_count, p_predict_tmp)
    p_predict_tmp.repartition(repartition, key='aid_pt', merge_func=merge, output=p_predict)

    return predict


def op_aggregate(truth, data, source, op=sum):
    getter = itemgetter(*data)
    agg = {int(tr): getter(source[tr]) for tr in truth}
    agg = {tr: int(op(val)) if isinstance(val, tuple) else int(val)
           for tr, val in agg.items()}
    return agg


def all_eval_score(df):
    source = defaultdict(Counter)
    for aid, aid2, score in iter_row_values(df.reset_index(), cols=['aid', 'aid2', 'score']):
        source[aid][aid2] = score

    test_labels = get_test(merge_test=True)
    data_iter = iter_row_values(test_labels, cols=[TRUTH_COL, 'data'])
    score_sum = [op_aggregate(truth, data, source, op=sum) for truth, data in data_iter]

    test_labels['aid'] = score_sum
    val = test_labels[['session', 'type', 'aid']].explode('aid')
    val['score'] = test_labels[['aid']].applymap(lambda x: x.values()).explode("aid").values
    return val


def all_eval_ranks(df):
    source = defaultdict(lambda: defaultdict(lambda: 10**10))
    for aid, aid2, rank in iter_row_values(df.reset_index(), cols=['aid', 'aid2', 'rank']):
        source[aid][aid2] = rank

    test_labels = get_test(merge_test=True)
    data_iter = iter_row_values(test_labels, cols=[TRUTH_COL, 'data'])
    rank_min = [op_aggregate(truth, data, source, op=min) for truth, data in data_iter]

    test_labels['aid'] = rank_min
    val = test_labels[['session', 'type', 'aid']].explode('aid')
    val['score'] = test_labels[['aid']].applymap(lambda x: x.values()).explode("aid").values
    return val


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
