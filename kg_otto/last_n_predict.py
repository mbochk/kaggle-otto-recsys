import pandas as pd


class LastNPredict:
    def predict(self, df):
        df = df.copy().drop_duplicates(['session', 'aid'], keep='last')
        df.drop('type', axis=1, inplace=True)
        df = df.groupby('session').tail(20)
        df = pd.concat([df] * 3, keys=[0, 1, 2], names=['type']) \
            .reset_index(level=0).sort_values(['session', 'type'], ignore_index=True)
        return df
