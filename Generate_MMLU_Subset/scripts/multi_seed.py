import numpy as np
import pandas as pd
from .subset_methods import (
    create_difficulty_bucket_subsets,
    create_irt_clustering_subsets,
    create_response_matrix_clustering_subsets,
)
from .evaluation import (
    compute_full_accuracy,
    compute_method_consistency,
    compute_method_representativeness,
    score_methods,
)


def run_multi_seed(
    seeds,
    Y,
    Y_train_mmlu,
    valid_indices,
    item_acc_valid,
    df_cn,
    excluded_subjects,
    irt_model_dir='data/irt_model/',
    subset_size=300,
):
    full_mmlu_acc = compute_full_accuracy(Y, valid_indices)

    per_seed_rows = []
    detail_per_seed = []
    for seed in seeds:
        subset1_bucket, subset2_bucket, _ = create_difficulty_bucket_subsets(
            item_acc_valid, valid_indices, subset_size, random_state=seed
        )
        subset1_irt, subset2_irt, _, _ = create_irt_clustering_subsets(
            valid_indices, subset_size, df_cn, excluded_subjects, irt_model_dir=irt_model_dir, random_state=seed
        )
        subset1_matrix, subset2_matrix, _, _ = create_response_matrix_clustering_subsets(
            Y_train_mmlu, valid_indices, subset_size, df_cn, excluded_subjects, random_state=seed
        )

        all_subsets_seed = {
            'subset1_bucket': subset1_bucket,
            'subset2_bucket': subset2_bucket,
            'subset1_irt': subset1_irt,
            'subset2_irt': subset2_irt,
            'subset1_matrix': subset1_matrix,
            'subset2_matrix': subset2_matrix,
        }

        subset_acc = {name: (Y[:, idxs].mean(axis=1)) for name, idxs in all_subsets_seed.items()}
        method_cons = compute_method_consistency(subset_acc)
        method_repr = compute_method_representativeness(subset_acc, full_mmlu_acc)
        method_scores = score_methods(method_cons, method_repr)
        ranking = sorted(method_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        best_method = ranking[0][0]

        for method in ['bucket', 'irt', 'matrix']:
            per_seed_rows.append({
                'seed': seed,
                '方法': method.upper(),
                '总分': method_scores[method]['total_score'],
                '与Full相关(均值)': method_repr[method]['avg_correlation'],
                '与Full均值差(均值)': method_repr[method]['avg_mean_diff'],
                '两子集相关': method_cons[method]['correlation'],
                '两子集均值差': method_cons[method]['mean_diff'],
                '两子集模型差(均值)': method_cons[method]['avg_model_diff'],
                '排名': 1 + [m for m, _ in ranking].index(method),
            })

        detail_per_seed.append({
            'seed': seed,
            'ranking': ranking,
            'best_method': best_method,
            'method_scores': method_scores,
            'method_consistency': method_cons,
            'method_representativeness': method_repr,
        })

    per_seed_df = pd.DataFrame(per_seed_rows)
    agg_rows = []
    for method in ['bucket', 'irt', 'matrix']:
        dfm = per_seed_df[per_seed_df['方法'] == method.upper()]
        wins = int((dfm['排名'] == 1).sum())
        agg_rows.append({
            '方法': method.upper(),
            '胜出次数/10': wins,
            '平均排名': float(dfm['排名'].mean()),
            '总分(均值)': float(dfm['总分'].mean()),
            '总分(Std)': float(dfm['总分'].std()),
            '与Full相关(均值)': float(dfm['与Full相关(均值)'].mean()),
            '与Full相关(Std)': float(dfm['与Full相关(均值)'].std()),
            '与Full均值差(均值)': float(dfm['与Full均值差(均值)'].mean()),
            '与Full均值差(Std)': float(dfm['与Full均值差(均值)'].std()),
            '两子集相关(均值)': float(dfm['两子集相关'].mean()),
            '两子集相关(Std)': float(dfm['两子集相关'].std()),
            '两子集均值差(均值)': float(dfm['两子集均值差'].mean()),
            '两子集均值差(Std)': float(dfm['两子集均值差'].std()),
            '两子集模型差(均值)': float(dfm['两子集模型差(均值)'].mean()),
            '两子集模型差(Std)': float(dfm['两子集模型差(均值)'].std()),
        })
    agg_df = pd.DataFrame(agg_rows).sort_values(['胜出次数/10', '总分(均值)'], ascending=[False, False])
    return per_seed_df, agg_df, detail_per_seed


