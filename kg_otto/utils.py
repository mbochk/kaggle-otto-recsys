from operator import itemgetter
import pandas as pd

ID_TO_TYPE = {
    0: "clicks",
    1: "carts",
    2: "orders"
}


def read_from_jsonl(data_path):
    df = pd.read_json(data_path, lines=True)
    df = df.explode('events')
    for col in ['aid', 'ts', 'type']:
        df[col] = df['events'].apply(itemgetter(col))
    df.drop('events', axis=1, inplace=True)
    return df


def pred_to_pred_list(pred:pd.DataFrame, col='aid'):
    pred = pred.sort_values(['session', 'type'], ignore_index=True)
    first = pred[['session', 'type']].drop_duplicates(keep='first')
    last = pred[['session', 'type']].drop_duplicates(keep='last')

    pred_vals = pred[col].values.tolist()
    aid_list = [pred_vals[start:end+1] for start, end in zip(first.index, last.index) ]
    first.reset_index(drop=True, inplace=True)
    first[col] = aid_list
    return first


def convert_to_listval(df: pd.DataFrame):
    if df.dtypes['type'] == 'uint8':
        df['type'] = df['type'].map(ID_TO_TYPE)
        df.sort_values(['session', 'type'], inplace=True)
    if not isinstance(df.aid.iloc[0], list):
        df = pred_to_pred_list(df, 'aid')
    return df


def do_eval(pred, truth):
    pass

