import numpy as np


def compute_full_accuracy(Y: np.ndarray, valid_indices):
    Y_valid = Y[:, valid_indices]
    full_acc = Y_valid.mean(axis=1)
    return full_acc


def compute_subset_accuracies(Y: np.ndarray, subsets: dict):
    subset_acc = {}
    for name, idxs in subsets.items():
        subset_acc[name] = Y[:, idxs].mean(axis=1)
    return subset_acc


def compute_method_consistency(subset_accuracies: dict):
    out = {}
    for method in ['bucket', 'irt', 'matrix']:
        a1 = subset_accuracies[f'subset1_{method}']
        a2 = subset_accuracies[f'subset2_{method}']
        diffs = np.abs(a1 - a2)
        out[method] = {
            'mean_diff': float(abs(a1.mean() - a2.mean())),
            'std_diff': float(abs(a1.std() - a2.std())),
            'correlation': float(np.corrcoef(a1, a2)[0, 1]),
            'avg_model_diff': float(diffs.mean()),
            'max_model_diff': float(diffs.max()),
        }
    return out


def compute_method_representativeness(subset_accuracies: dict, full_acc):
    out = {}
    for method in ['bucket', 'irt', 'matrix']:
        a1 = subset_accuracies[f'subset1_{method}']
        a2 = subset_accuracies[f'subset2_{method}']
        corr1 = float(np.corrcoef(a1, full_acc)[0, 1])
        corr2 = float(np.corrcoef(a2, full_acc)[0, 1])
        diff1 = float(abs(a1.mean() - full_acc.mean()))
        diff2 = float(abs(a2.mean() - full_acc.mean()))
        std1 = float(abs(a1.std() - full_acc.std()))
        std2 = float(abs(a2.std() - full_acc.std()))
        out[method] = {
            'avg_correlation': (corr1 + corr2) / 2,
            'avg_mean_diff': (diff1 + diff2) / 2,
            'avg_std_diff': (std1 + std2) / 2,
        }
    return out


def score_methods(method_consistency: dict, method_repr: dict, weights=None):
    if weights is None:
        weights = {'consistency': 0.5, 'representativeness': 0.5}
    scores = {}
    for method in ['bucket', 'irt', 'matrix']:
        c = method_consistency[method]
        r = method_repr[method]
        consistency_score = (
            (1 / (1 + c['mean_diff'])) * 0.3 +
            (1 / (1 + c['avg_model_diff'])) * 0.3 +
            c['correlation'] * 0.4
        )
        repr_score = (
            r['avg_correlation'] * 0.5 +
            (1 / (1 + r['avg_mean_diff'])) * 0.3 +
            (1 / (1 + r['avg_std_diff'])) * 0.2
        )
        total = consistency_score * weights['consistency'] + repr_score * weights['representativeness']
        scores[method] = {
            'consistency_score': float(consistency_score),
            'representativeness_score': float(repr_score),
            'total_score': float(total),
        }
    return scores


