import logging
import os
import shutil
import pandas as pd

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Optional

from kg_otto.iter import iter_tqdm


@dataclass
class PartitionedDataFrame:
    path: str
    parallelism: int = 20
    use_timer: bool = True

    def create(self, if_not_exists=True):
        exists = os.path.isdir(self.path)
        if exists and not if_not_exists:
            raise ValueError(f"Directory {self.path} already exists")
        if not exists:
            os.mkdir(self.path)
        return self

    def remove(self, pt=None):
        path = self.path if not pt else self.path + '/' + pt
        shutil.rmtree(path, ignore_errors=True)
        return self

    def write(self, df: pd.DataFrame, pt=None):
        path = self.path if not pt else self.path + '/' + pt
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path)

    @property
    def partitions(self):
        return sorted(os.listdir(self.path))

    def _run_mp_pool(self, func):
        data = self.partitions
        with Pool(processes=self.parallelism) as pool:
            iter_res = pool.imap_unordered(func, data)
            if self.use_timer:
                res = iter_tqdm(iter_res, total=len(data), smoothing=0)
            else:
                res = list(iter_res)
        return res

    def get_df(self, pt=None):
        path = self.path
        if pt:
            path = path + '/' + pt
        return pd.read_parquet(path)

    def mp_apply(self, func: Callable, output: 'PartitionedDataFrame'):
        new_func = PartitionedDecorator(self, output, func)
        self._run_mp_pool(new_func)

    def repartition(
            self, map_func: Callable, output: 'PartitionedDataFrame',
            key: str, merge_func: Optional[Callable]
    ):
        map_func = RepartitionDecorator(self, output, map_func, key)
        logging.info("Do shuffle maps")
        self._run_mp_pool(map_func)
        merge_func = MergeDecorator(output, merge_func)
        logging.info("Do merge reduces")
        output._run_mp_pool(merge_func)


@dataclass
class PartitionedDecorator:
    input: PartitionedDataFrame
    output: PartitionedDataFrame
    func: Callable

    def __call__(self, pt: str):
        res_df = self.func(self.input.get_df(pt))
        self.output.write(res_df, pt)


@dataclass
class RepartitionDecorator:
    input: PartitionedDataFrame
    output: PartitionedDataFrame
    func: Callable
    partition_key: str

    def __call__(self, pt: str):
        df = self.input.get_df(pt)
        df: pd.DataFrame = self.func(df)
        assert self.partition_key in df.columns

        key_partitions = []
        for key, grp in df.groupby(self.partition_key):
            key_pt = f'{self.partition_key}={key}'
            new_pt = f'{key_pt}/{pt}'
            self.output.write(grp, new_pt)
            key_partitions.append(key_pt)
        return key_partitions


@dataclass
class MergeDecorator:
    input: PartitionedDataFrame
    func: Optional[Callable] = None

    def __call__(self, pt):
        pt_df = self.input.get_df(pt)
        if self.func:
            pt_df = self.func(pt_df)
        self.input.remove(pt)
        self.input.write(pt_df, pt)
