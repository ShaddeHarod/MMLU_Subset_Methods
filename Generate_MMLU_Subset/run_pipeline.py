import os
import json
import argparse
import numpy as np
import pandas as pd

from scripts.utils import scenarios, prepare_data, create_responses
from scripts.data_io import load_leaderboard_pickle, load_mmlu_zh_cn_csv, export_subset_sorted, ensure_dir, save_json
from scripts.subset_methods import (
    create_difficulty_bucket_subsets,
    create_irt_clustering_subsets,
    create_response_matrix_clustering_subsets,
)
from scripts.evaluation import (
    compute_full_accuracy,
    compute_subset_accuracies,
    compute_method_consistency,
    compute_method_representativeness,
    score_methods,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate MMLU subsets and evaluate methods.')
    parser.add_argument('--lb_pickle', default='data/lb.pickle', help='Leaderboard pickle path')
    parser.add_argument('--mmlu_csv', default='mmlu_ZH-CN.csv', help='MMLU ZH-CN csv path')
    parser.add_argument('--irt_model_dir', default='data/irt_model/', help='IRT model directory (with best_parameters.json)')
    parser.add_argument('--subset_size', type=int, default=300, help='Subset size per split')
    parser.add_argument('--out_dir_csv', default='mmlu_subsets_csv', help='Output directory for subset CSVs')
    parser.add_argument('--save_summary', default='mmlu_ZH-CN_subset_summary.json', help='Summary JSON filename (root)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    data = load_leaderboard_pickle(args.lb_pickle)
    scenarios_position, subscenarios_position = prepare_data(scenarios, data)
    Y = create_responses(scenarios, data)

    # MMLU slice and per-item difficulty proxy
    Y_mmlu = Y[:, scenarios_position['mmlu']]
    item_acc = Y_mmlu.mean(axis=0)

    # Load Chinese CSV and filter excluded subjects
    df_cn = load_mmlu_zh_cn_csv(args.mmlu_csv)
    if df_cn.shape[0] != item_acc.shape[0]:
        raise ValueError('Row mismatch between mmlu_ZH-CN.csv and MMLU items in responses.')

    excluded_subjects = {
        'high_school_us_history', 'security_studies', 'high_school_government_and_politics',
        'jurisprudence', 'business_ethics', 'us_foreign_policy', 'global_facts', 'moral_scenarios',
        'professional_law', 'moral_disputes'
    }
    mask_keep = ~df_cn['Subject'].isin(excluded_subjects)
    valid_indices = df_cn[mask_keep].index.values

    Y_mmlu_valid = Y_mmlu[:, valid_indices]
    item_acc_valid = item_acc[valid_indices]

    # Generate subsets by three methods
    subset1_bucket, subset2_bucket, _ = create_difficulty_bucket_subsets(
        item_acc_valid, valid_indices, args.subset_size, random_state=args.random_state
    )

    subset1_irt, subset2_irt, _, _ = create_irt_clustering_subsets(
        valid_indices, args.subset_size, df_cn, excluded_subjects, irt_model_dir=args.irt_model_dir, random_state=args.random_state
    )

    # For matrix-based clustering use all models as training by default
    subset1_matrix, subset2_matrix, _, _ = create_response_matrix_clustering_subsets(
        Y_mmlu, valid_indices, args.subset_size, df_cn, excluded_subjects, random_state=args.random_state
    )

    all_subsets = {
        'subset1_bucket': subset1_bucket,
        'subset2_bucket': subset2_bucket,
        'subset1_irt': subset1_irt,
        'subset2_irt': subset2_irt,
        'subset1_matrix': subset1_matrix,
        'subset2_matrix': subset2_matrix,
    }

    # Evaluate
    full_mmlu_acc = compute_full_accuracy(Y, valid_indices)
    subset_accuracies = compute_subset_accuracies(Y, all_subsets)
    method_consistency = compute_method_consistency(subset_accuracies)
    method_repr = compute_method_representativeness(subset_accuracies, full_mmlu_acc)
    method_scores = score_methods(method_consistency, method_repr)
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
    best_method = sorted_methods[0][0]

    # Export CSVs
    cols = ['ID', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Subject']
    ensure_dir(args.out_dir_csv)
    export_paths = {}
    for name, idxs in all_subsets.items():
        filename = f"mmlu_ZH-CN_{name}.csv"
        path = os.path.join(args.out_dir_csv, filename)
        export_subset_sorted(df_cn, idxs, path, cols)
        export_paths[name] = path

    # Save summary
    summary_payload = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'subset_size': args.subset_size,
        'random_state': args.random_state,
        'best_method': best_method,
        'method_scores': method_scores,
        'method_consistency': method_consistency,
        'method_representativeness': method_repr,
        'all_subsets': {k: list(map(int, v.tolist() if hasattr(v, 'tolist') else list(v))) for k, v in all_subsets.items()},
        'export_paths': export_paths,
    }
    save_json(args.save_summary, summary_payload)

    # Print brief report
    print('Exported subset CSVs:')
    for name, path in export_paths.items():
        print(f'- {name}: {path}')
    print(f"Best method: {best_method.upper()} (score={method_scores[best_method]['total_score']:.4f})")
    print(f"Summary saved to: {args.save_summary}")


if __name__ == '__main__':
    main()


