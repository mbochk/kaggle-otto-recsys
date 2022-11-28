import pandas as pd
from dataclasses import dataclass


@dataclass
class LastNPredict:
    # prioritize clicks more than carts more than orders rather than time
    reverse_type_sort: bool = False
    # do not limit yourself with top20 to get
    unfair: bool = False
    keep: str = 'last'

    def predict(self, df):
        df = df.copy().drop_duplicates(['session', 'aid'], keep=self.keep)
        if self.reverse_type_sort:
            df = df.sort_values(['session', 'type', 'type'], ascending=[True, True, False])
        df.drop('type', axis=1, inplace=True)
        if not self.unfair:

            df = df.groupby('session').tail(20)
        df = pd.concat([df] * 3, keys=[0, 1, 2], names=['type']) \
            .reset_index(level=0).sort_values(['session', 'type'], ignore_index=True)
        return df

