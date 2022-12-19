from kg_otto import set_log_level, pd, np
from kg_otto.utils import read_test_labels, do_eval
from pathlib import Path
import logging


set_log_level(20)


def get_data():
    data_path = Path("/home/mikhail.bochkarev/study/")

    logging.info("load train")
    train_df = pd.read_parquet(data_path / "split_train.parquet")

    logging.info("load test")
    test_df = pd.read_parquet(data_path / "test.parquet")
    test_labels = read_test_labels(data_path / "test_labels.parquet")
    return train_df, test_df, test_labels
