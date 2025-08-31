import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .irt import load_irt_parameters


def create_difficulty_bucket_subsets(item_difficulties, valid_indices, subset_size, num_buckets=5, random_state=42):
    rng = np.random.RandomState(random_state)
    bucket_labels = pd.qcut(item_difficulties, q=num_buckets, labels=False, duplicates='drop')
    if pd.isna(bucket_labels).any():
        ranks = pd.Series(item_difficulties).rank(method='average') / len(item_difficulties)
        bucket_labels = np.minimum((ranks * num_buckets).astype(int), num_buckets - 1)

    bucket_counts = pd.Series(bucket_labels).value_counts().sort_index()
    bucket_props = bucket_counts / bucket_counts.sum()
    target_counts_float = bucket_props * subset_size
    target_counts = np.floor(target_counts_float).astype(int)
    remainder = subset_size - target_counts.sum()
    if remainder > 0:
        remainder_probs = target_counts_float - target_counts
        remainder_buckets = remainder_probs.nlargest(remainder).index
        target_counts[remainder_buckets] += 1

    subset1_indices = []
    subset2_indices = []
    for bucket_id in sorted(bucket_counts.index):
        bucket_mask = bucket_labels == bucket_id
        bucket_items = np.where(bucket_mask)[0]
        target_count = target_counts[bucket_id]
        if target_count > 0 and len(bucket_items) >= target_count * 2:
            selected = rng.choice(bucket_items, size=target_count * 2, replace=False)
            rng.shuffle(selected)
            subset1_indices.extend(selected[:target_count])
            subset2_indices.extend(selected[target_count:target_count * 2])
        elif target_count > 0:
            rng.shuffle(bucket_items)
            mid = len(bucket_items) // 2
            subset1_indices.extend(bucket_items[:mid])
            subset2_indices.extend(bucket_items[mid:])

    all_selected = set(subset1_indices + subset2_indices)
    remaining_items = [i for i in range(len(item_difficulties)) if i not in all_selected]
    while len(subset1_indices) < subset_size and remaining_items:
        idx = rng.choice(len(remaining_items))
        subset1_indices.append(remaining_items.pop(idx))
    while len(subset2_indices) < subset_size and remaining_items:
        idx = rng.choice(len(remaining_items))
        subset2_indices.append(remaining_items.pop(idx))

    subset1_original = valid_indices[subset1_indices[:subset_size]]
    subset2_original = valid_indices[subset2_indices[:subset_size]]
    return subset1_original, subset2_original, bucket_labels


def create_irt_clustering_subsets(valid_indices, subset_size, df_cn, excluded_subjects, irt_model_dir='data/irt_model/', random_state=42):
    rng = np.random.RandomState(random_state)
    A, B, _ = load_irt_parameters(irt_model_dir)
    irt_features = np.vstack((A.squeeze(), B.squeeze())).T
    irt_features_valid = irt_features[valid_indices]
    kmeans = KMeans(n_clusters=subset_size, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(irt_features_valid)
    cluster_centers = kmeans.cluster_centers_

    subset1_indices = []
    subset2_indices = []
    subjects_valid = df_cn.loc[valid_indices, 'Subject'].values

    for cluster_id in range(subset_size):
        cluster_items = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_items) == 0:
            continue
        cluster_features = irt_features_valid[cluster_items]
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_features - center, axis=1)
        sorted_indices = np.argsort(distances)
        valid_items = []
        for idx in sorted_indices:
            item_idx = cluster_items[idx]
            item_subject = subjects_valid[item_idx]
            if item_subject not in excluded_subjects:
                valid_items.append(item_idx)
            if len(valid_items) >= 2:
                break
        if len(valid_items) >= 2:
            rng.shuffle(valid_items)
            subset1_indices.append(valid_items[0])
            subset2_indices.append(valid_items[1])
        elif len(valid_items) == 1:
            if rng.random() < 0.5:
                subset1_indices.append(valid_items[0])
            else:
                subset2_indices.append(valid_items[0])

    while len(subset1_indices) < subset_size and len(subset2_indices) > 0:
        idx = rng.choice(len(subset2_indices))
        subset1_indices.append(subset2_indices.pop(idx))
    while len(subset2_indices) < subset_size and len(subset1_indices) > subset_size:
        idx = rng.choice(len(subset1_indices[subset_size:]))
        subset2_indices.append(subset1_indices.pop(subset_size + idx))

    subset1_original = valid_indices[subset1_indices[:subset_size]]
    subset2_original = valid_indices[subset2_indices[:subset_size]]
    return subset1_original, subset2_original, cluster_labels, cluster_centers


def create_response_matrix_clustering_subsets(Y_train, valid_indices, subset_size, df_cn, excluded_subjects, random_state=42):
    rng = np.random.RandomState(random_state)
    Y_mmlu_train = Y_train[:, :]  # caller should pass already-sliced MMLU if needed
    Y_mmlu_train_valid = Y_mmlu_train[:, valid_indices]
    response_matrix = Y_mmlu_train_valid.T
    kmeans = KMeans(n_clusters=subset_size, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(response_matrix)
    cluster_centers = kmeans.cluster_centers_

    subset1_indices = []
    subset2_indices = []
    subjects_valid = df_cn.loc[valid_indices, 'Subject'].values

    for cluster_id in range(subset_size):
        cluster_items = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_items) == 0:
            continue
        cluster_responses = response_matrix[cluster_items]
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_responses - center, axis=1)
        sorted_indices = np.argsort(distances)
        valid_items = []
        for idx in sorted_indices:
            item_idx = cluster_items[idx]
            item_subject = subjects_valid[item_idx]
            if item_subject not in excluded_subjects:
                valid_items.append(item_idx)
            if len(valid_items) >= 2:
                break
        if len(valid_items) >= 2:
            rng.shuffle(valid_items)
            subset1_indices.append(valid_items[0])
            subset2_indices.append(valid_items[1])
        elif len(valid_items) == 1:
            if rng.random() < 0.5:
                subset1_indices.append(valid_items[0])
            else:
                subset2_indices.append(valid_items[0])

    while len(subset1_indices) < subset_size and len(subset2_indices) > 0:
        idx = rng.choice(len(subset2_indices))
        subset1_indices.append(subset2_indices.pop(idx))
    while len(subset2_indices) < subset_size and len(subset1_indices) > subset_size:
        idx = rng.choice(len(subset1_indices[subset_size:]))
        subset2_indices.append(subset1_indices.pop(subset_size + idx))

    subset1_original = valid_indices[subset1_indices[:subset_size]]
    subset2_original = valid_indices[subset2_indices[:subset_size]]
    return subset1_original, subset2_original, cluster_labels, cluster_centers


