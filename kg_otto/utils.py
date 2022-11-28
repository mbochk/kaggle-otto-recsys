from operator import itemgetter
import pandas as pd

TYPE_TO_ID = {
    "clicks": 0,
    "carts": 1,
    "orders": 2
}

ID_TO_TYPE = {v: k for k, v in TYPE_TO_ID.items()}


def iter_row_values(df, cols=None):
    cols = cols or list(df.columns)
    values = [df[col] for col in cols]
    yield from zip(*values)


def read_from_jsonl(data_path):
    df = pd.read_json(data_path, lines=True)
    df = df.explode('events')
    for col in ['aid', 'ts', 'type']:
        df[col] = df['events'].apply(itemgetter(col))
    df.drop('events', axis=1, inplace=True)
    return df


def read_test_labels(data_path, src='parquet'):
    if src == 'parquet':
        df = pd.read_parquet(data_path)
        df['ground_truth'] = df['ground_truth'].apply(set)
        df['ground_truth_len'] = df['ground_truth'].apply(len)
        df['type'] = df['type'].map(TYPE_TO_ID).astype('uint8')
    else:
        raise ValueError
    return df


def pred_to_pred_list(pred: pd.DataFrame, col='aid'):
    pred = pred.sort_values(['session', 'type'], ignore_index=True)
    first = pred[['session', 'type']].drop_duplicates(keep='first')
    last = pred[['session', 'type']].drop_duplicates(keep='last')

    pred_vals = pred[col].values.tolist()
    aid_list = [pred_vals[start:end+1] for start, end in zip(first.index, last.index) ]
    first.reset_index(drop=True, inplace=True)
    first[col] = aid_list
    return first


def convert_to_listval(df: pd.DataFrame):
    if df.dtypes['type'] == 'O':
        df['type'] = df['type'].map(TYPE_TO_ID)
        df.sort_values(['session', 'type'], inplace=True)
    if 'aid' in df.columns and not isinstance(df.aid.iloc[0], list):
        df = pred_to_pred_list(df, 'aid')
    return df


def do_eval(pred, test_labels):
    merged = pd.merge_ordered(test_labels, convert_to_listval(pred), how='left', on=['session', 'type'])

    it = iter_row_values(merged, ['ground_truth', 'aid'])
    hits = pd.Series([len(y_true.intersection(y_pred)) for y_true, y_pred in it], index=merged.index)

    recall = hits.groupby(merged.type).sum() / merged.ground_truth_len.clip(upper=20).groupby(merged.type).sum()
    score = (recall * pd.Series([0.10, 0.30, 0.60])).sum()
    recall[3] = score
    return recall


def pred_to_submission(pred, output_path):
    pred = pred_to_pred_list(pred)
    ss = pred.session.astype('string') + '_' + pred.type.map(ID_TO_TYPE)
    labels = pred.aid.apply(lambda x: ' '.join(map(str, x)))
    pd.concat([ss, labels], axis=1, keys=['session_type', 'labels']).to_csv(output_path)
    print(output_path)


def submission_to_pred(path):
    df = pd.read_csv(path, sep=',')[['session_type', 'labels']]
    vals = [[ss.split('_'), ll.split(' ')] for ss, ll in iter_row_values(df)]
    vals = [[int(ss[0]), TYPE_TO_ID[ss[1]], ll] for ss, ll in vals]
    df = pd.DataFrame(vals, columns=['session', 'type', 'aid'])
    df.sort_values(['session', 'type'], inplace=True, ignore_index=True)
    return df
