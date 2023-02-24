import os

from kg_otto.partitioned import PartitionedDataFrame
from kg_otto.imp.i2i_max import all_i2i_count


def main():
    train = "partitioned_train.parquet"
    predict = 'partitioned_predict_i2i_full.parquet'

    p_train = PartitionedDataFrame(train)
    if predict in os.listdir():
        return
    else:
        os.mkdir(predict)
        output = PartitionedDataFrame(predict)

    p_train.mp_apply(all_i2i_count, output)


if __name__ == "__main__":
    main()
