from kg_otto import *
from kg_otto.utils import read_test_labels
from pathlib import Path

data_path = Path("/home/mikhail.bochkarev/study/")

train_df = pd.read_parquet(data_path/"split_train.parquet")

test_df = pd.read_parquet(data_path/"test.parquet")
test_labels = read_test_labels(data_path/"test_labels.parquet")

from kg_otto.last_n_predict import LastNPredict
from kg_otto.utils import do_eval

ll = LastNPredict(reverse_type_sort=True, keep='last')

pred = ll.predict(test_df)

print(do_eval(pred, test_labels))
