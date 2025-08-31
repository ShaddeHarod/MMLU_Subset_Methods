import numpy as np


def compute_balance_weights(Y: np.ndarray, scenarios_position: dict, subscenarios_position: dict, scenarios: dict) -> np.ndarray:
    """
    Compute per-item balance weights so that subscenarios contribute equally within scenarios.

    Only MMLU has multiple subscenarios in this project, but the function is generic.
    """
    balance_weights = np.ones(Y.shape[1])
    for scenario in scenarios.keys():
        N = len(scenarios_position[scenario])
        n_sub = len(scenarios[scenario])
        for sub in scenarios[scenario]:
            n_i = len(subscenarios_position[scenario][sub])
            balance_weights[subscenarios_position[scenario][sub]] = N / (n_sub * n_i)
    return balance_weights


