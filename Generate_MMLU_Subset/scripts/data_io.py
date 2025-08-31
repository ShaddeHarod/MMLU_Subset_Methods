import os
import json
import pickle
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_leaderboard_pickle(path: str):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def load_mmlu_zh_cn_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_subset_sorted(df_source: pd.DataFrame, indices, path: str, cols, sort_by=('Subject', 'ID'), ascending=(True, True)) -> None:
    df_sub = df_source.loc[indices, cols].copy()
    # stable sort: Subject asc, ID asc to match prior behavior
    df_sub = df_sub.sort_values(by=list(sort_by), ascending=list(ascending), kind='mergesort')
    ensure_dir(os.path.dirname(path) or '.')
    df_sub.to_csv(path, index=False, encoding='utf-8')


