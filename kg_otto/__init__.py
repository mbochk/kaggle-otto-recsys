import logging
import pandas as pd
import numpy as np
from kg_otto.config import SCORE_COL, ITEM_COL, SESSION_COL
from kg_otto.utils import read_test_labels, do_eval
from pathlib import Path


from kg_otto.data import get_data, get_test
from kg_otto.partitioned import PartitionedDataFrame
from .utils import pred_to_pred_list, do_eval


def set_log_level(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]–[%(levelname)s]–[%(message)s](%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


try:
    ipython = get_ipython()
    ipython.run_line_magic("load_ext", 'autoreload')
    ipython.run_line_magic("autoreload", "2")
    ipython.run_line_magic("matplotlib", "inline")
    ipython.run_line_magic('config', 'Completer.use_jedi = False')

    pd.options.display.max_rows = 40
    pd.options.display.min_rows = 20

    import seaborn
    seaborn.set_theme(style='whitegrid')
    set_log_level(20)
except NameError:
    pass


