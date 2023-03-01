import pandas as pd
from dataclasses import dataclass

from kg_otto.data import get_test, DEFAULT_PATH


@dataclass
class I2iDatasetJoiner:
    target_type: int
    test_path: str = DEFAULT_PATH

    def get_test(self):
        test_df, labels = get_test(path=self.test_path, merge_test=False)
        test_df = test_df[['session', 'aid']].drop_duplicates()
        labels = labels[labels.type == 0][
            ['session', 'truth']
        ].explode('truth')
        labels['target'] = 1
        labels.rename({"truth": "aid2"}, axis=1, inplace=True)
        return test_df, labels

    def __call__(self, df):
        if not df.aid.is_monotonic_increasing:
            df = df.sort_valus('aid', inplace=True)

        test_df, labels = self.get_test()

        # join data
        test_df = test_df[test_df.aid.isin(df.aid)].sort_values('aid')
        df = pd.merge_ordered(df, test_df, how='inner')

        # aggregate features
        df = df.groupby(['session', 'aid2'], as_index=False).agg({
            "score": "sum", "rank": "min"
        })

        # filter by features

        # join truth
        df = pd.merge(df, labels, on=['session', 'aid2'], how='left')
        df['target'] = df['target'].fillna(0)
        df.sort_values('session', inplace=True)
        df.rename({"aid2": "aid"}, axis=1, inplace=True)
        return df
