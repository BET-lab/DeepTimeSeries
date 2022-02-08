import itertools

import pandas as pd


def logical_and_for_set_list(set_list):
    s = set_list[0]
    for s_ in set_list[1:]:
        s = s & s_
    return s


def logical_or_for_set_list(set_list):
    s = set_list[0]
    for s_ in set_list[1:]:
        s = s | s_
    return s


def merge_dicts(dicts, ignore_keys=None):
    # Check key duplication.
    assert len(
        logical_and_for_set_list([set(d.keys()) for d in dicts])
    ) == 0, 'Keys of dictionaries are duplicated.'

    if ignore_keys is None:
        ignore_keys = set()
    else:
        ignore_keys = set(ignore_keys)

    merged_dict = {}
    for k, v in itertools.chain(*[d.items() for d in dicts]):
        if k in ignore_keys:
            continue
        merged_dict[k] = v

    return merged_dict


def merge_data_frames(dfs):
    dfs = dfs.copy()
    for i, df in enumerate(dfs):
        df['time_index'] = df.index
        df['time_series_id'] = i

    return pd.concat(dfs).reset_index(drop=True)