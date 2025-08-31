import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from .irt import load_irt_parameters


def compute_anchor_points_and_weights(
    Y_train: np.ndarray,
    scenarios: dict,
    scenarios_position: dict,
    balance_weights: np.ndarray,
    subscenarios_position: dict,
    number_item: int,
    clustering: str = 'correct.',
    irt_model_dir: str = 'data/irt_model/',
    random_state: int = 42,
):
    anchor_points = {}
    anchor_weights = {}

    for scenario in scenarios.keys():
        if clustering == 'correct.':
            X = Y_train[:, scenarios_position[scenario]].T
        elif clustering == 'irt':
            A, B, _ = load_irt_parameters(irt_model_dir)
            X_all = np.vstack((A.squeeze(), B.squeeze().reshape((1, -1)))).T
            X = X_all[scenarios_position[scenario]]
        else:
            raise NotImplementedError('Unknown clustering mode')

        norm_balance_weights = balance_weights[scenarios_position[scenario]].copy()
        norm_balance_weights /= norm_balance_weights.sum()

        kmeans = KMeans(n_clusters=number_item, n_init='auto', random_state=random_state)
        kmeans.fit(X, sample_weight=norm_balance_weights)

        anchor_points[scenario] = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean').argmin(axis=1)
        anchor_weights[scenario] = np.array([
            np.sum(norm_balance_weights[kmeans.labels_ == c]) for c in range(number_item)
        ])

    return anchor_points, anchor_weights


def evaluate_anchor_estimation(
    Y_test: np.ndarray,
    scenarios: dict,
    scenarios_position: dict,
    anchor_points: dict,
    anchor_weights: dict,
    balance_weights: np.ndarray,
):
    results = {}
    for scenario in scenarios.keys():
        Y_anchor = Y_test[:, scenarios_position[scenario]][:, anchor_points[scenario]]
        Y_hat = (Y_anchor * anchor_weights[scenario]).sum(axis=1)
        Y_true = (balance_weights * Y_test)[:, scenarios_position[scenario]].mean(axis=1)
        results[scenario] = float(np.abs(Y_hat - Y_true).mean())
    return results


