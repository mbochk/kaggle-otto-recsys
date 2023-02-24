from kg_otto import set_log_level, pd
from kg_otto.utils import read_test_labels
from pathlib import Path
import logging


DEFAULT_PATH = "/home/mikhail.bochkarev/study/"


def merge_test_data(test_df, test_labels):
    test_data = test_df.groupby("session").aid.apply(set)
    test_data = test_data.reindex(test_labels.session)
    test_labels['data'] = test_data.values


def get_test(path=DEFAULT_PATH, merge_test=True):
    data_path = Path(path)
    test_df = pd.read_parquet(data_path / "test.parquet")
    test_labels = read_test_labels(data_path / "test_labels.parquet")

    if merge_test:
        merge_test_data(test_df, test_labels)
        return test_labels
    else:
        return test_df, test_labels


def get_data(path=DEFAULT_PATH, merge_test=False):
    set_log_level(20)
    data_path = Path(path)

    logging.info("load train")
    train_df = pd.read_parquet(data_path / "split_train.parquet")

    logging.info("load test")
    test_df = get_test(path, merge_test)

    return train_df, test_df
