import numpy as np
from scipy.optimize import minimize
import jsonlines
import os
import json
import time
from .utils import item_curve


def create_irt_dataset(responses, dataset_name):
    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}
        for j in range(responses.shape[1]):
            aux_q['q' + str(j)] = int(responses[i, j])
        aux['subject_id'] = str(i)
        aux['responses'] = aux_q
        dataset.append(aux)
    with jsonlines.open(dataset_name, mode='w') as writer:
        writer.write_all([dataset[i] for i in range(len(dataset))])


def train_irt_model(dataset_name, model_name, D, lr, epochs, device):
    command = f"py-irt train 'multidim_2pl' {dataset_name} {model_name} --dims {D} --lr {lr} --epochs {epochs} --device {device} --priors 'hierarchical' --seed 42 --deterministic --log-every 200"
    os.system(command)


def load_irt_parameters(model_name):
    with open(model_name + 'best_parameters.json') as f:
        params = json.load(f)
    A = np.array(params['disc']).T[None, :, :]
    B = np.array(params['diff']).T[None, :, :]
    Theta = np.array(params['ability'])[:, :, None]
    return A, B, Theta


def estimate_ability_parameters(responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]

    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A, B).squeeze()
        log_likelihood = np.sum(responses_test * np.log(P + eps) + (1 - responses_test) * np.log(1 - P + eps))
        return -log_likelihood

    if type(theta_init) == np.ndarray:
        theta_init = theta_init.reshape(-1)
        assert theta_init.shape[0] == D
    else:
        theta_init = np.zeros(D)

    optimal_theta = minimize(neg_log_like, theta_init, method=optimizer).x[None, :, None]
    return optimal_theta


