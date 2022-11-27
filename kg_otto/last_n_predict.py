class LastNPredict:
    def predict(self, df):
        df = df.drop_duplicates(['session', 'aid'], keep='last')
        df = df.groupby('session').tail(20)
        return df