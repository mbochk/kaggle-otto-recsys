import pandas as pd
from .utils import pred_to_pred_list, do_eval


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
except NameError:
    pass
