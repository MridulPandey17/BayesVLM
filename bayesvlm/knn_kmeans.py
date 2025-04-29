# knn.py

import torch
from tqdm import tqdm
from collections import OrderedDict
from bayesvlm.vlm import EncoderResult

def diagonal_wasserstein_distance(mu1, mu2, cov1, cov2):
    """
    Computes the squared diagonal Wasserstein distance between sets of distributions.
    Assumes covariance matrices are diagonal.

    Args:
        mu1 (Tensor): Means of the first set of distributions [N, D].
        mu2 (Tensor): Means of the second set of distributions [M, D].
        cov1 (Tensor): Diagonal variances of the first set [N, D].
        cov2 (Tensor): Diagonal variances of the second set [M, D].

    Returns:
        Tensor: Pairwise squared diagonal Wasserstein distances [N, M].
    """
    # Compute L2 squared distance between means
    l2_squared = torch.cdist(mu1, mu2)**2 # [N, M]

    # Compute product term for variances
    # Need sqrt(cov1) [N, D] and sqrt(cov2) [M, D]
    # Einstein sum: einsum('nd,md->nm', sqrt(cov1), sqrt(cov2)) computes N*M pairwise dot products
    # We need 2 * sqrt(cov1_i) * sqrt(cov2_j) summed over d: 2 * sum_d(sqrt(cov1_nd) * sqrt(cov2_md))
    # Correct Einstein sum: 'ak,bk->ab' computes sum_{k} (a_ak * b_bk) resulting in [N, M]
    var_prod = 2 * torch.einsum('nd,md->nm', torch.sqrt(cov1), torch.sqrt(cov2)) #[N, M]

    # Compute sum of variances across dimensions for each distribution
    sum_var1 = cov1.sum(dim=-1) # [N]
    sum_var2 = cov2.sum(dim=-1) # [M]

    # Add singleton dimensions for broadcasting: sum_var1 becomes [N, 1], sum_var2 becomes [1, M]
    sum_var1 = sum_var1.unsqueeze(1)
    sum_var2 = sum_var2.unsqueeze(0)

    # Compute the diagonal Wasserstein distance^2
    # distance(i, j) = ||mu1_i - mu2_j||^2 + sum_d(var1_id + var2_jd - 2*sqrt(var1_id * var2_jd))
    # which simplifies to ||mu1_i - mu2_j||^2 + sum_d(var1_id) + sum_d(var2_jd) - 2 * sum_d(sqrt(var1_id) * sqrt(var2_jd))
    diagonal_wasserstein_sq = l2_squared + sum_var1 + sum_var2 - var_prod

    # Ensure non-negativity due to potential numerical issues
    diagonal_wasserstein_sq = torch.clamp(diagonal_wasserstein_sq, min=0.0)

    return diagonal_wasserstein_sq

def wdist2(mu1, mu2, cov1, cov2):
    """ Alias for diagonal_wasserstein_distance (squared). """
    return diagonal_wasserstein_distance(mu1, mu2, cov1, cov2)

def _remove_last_elements_to_keep_n_unique(indices: torch.Tensor, n):
    """ Helper function to truncate a tensor until it contains exactly n unique elements. """
    # Make a copy to avoid modifying the original tensor if it's passed around
    current_indices = indices.clone()
    while len(torch.unique(current_indices)) > n:
        current_indices = current_indices[:-1]
    # Check if we undershot (unlikely with typical buffer sizes but possible)
    if len(torch.unique(current_indices)) < n:
         print(f"Warning: _remove_last_elements_to_keep_n_unique ended with {len(torch.unique(current_indices))} unique elements, requested {n}. Using available unique elements.")
         # In this case, we return what we have which has <= n unique elements
    return current_indices

def extract_test_train_indices(text_idx_to_train_data):
    """ Extracts unique test and train indices from the k-NN mapping structure. """
    test_indices = []
    train_indices_flat = []
    for test_idx, data in text_idx_to_train_data.items():
        test_indices.append(int(test_idx))
        # Ensure indices are integers before extending
        train_indices_flat.extend([int(x) for x in data['indices']])

    # Remove duplicates by converting to a set and back to a list
    train_indices = list(OrderedDict.fromkeys(train_indices_flat).keys())

    return dict(test=test_indices, train=train_indices)

def find_similar_samples_cosine(
    train: EncoderResult,                 # Features of the representative training subset
    test: EncoderResult,                  # Features of the full test set
    indices_test: torch.Tensor,           # Indices of the uncertain test samples
    values_test: torch.Tensor,            # Scores/values of the uncertain test samples
    original_train_indices: torch.Tensor, # Original indices (in full train set) of the representatives
    k_nearest: int,
    source_covariance,
    device: str,
    buffersize=150,                       # Buffer for ensuring unique neighbours
):
    """
    Finds k_nearest neighbors in the representative training subset using Expected Cosine Similarity
    for given uncertain test samples, mapping back to original training indices.

    Args:
        train: EncoderResult for the representative training samples.
        test: EncoderResult for the full test set.
        indices_test: Indices (within 'test') of uncertain samples to find neighbors for.
        values_test: Acquisition scores for the uncertain test samples.
        original_train_indices: Indices (within the original full training set) corresponding
                                to the samples in 'train'.
        k_nearest: Number of nearest neighbors required per test sample.
        source_covariance: Covariance object for source embeddings (needed for expected norm).
        device: Device to run the computations on.
        buffersize: How many extra neighbors to fetch initially to handle non-unique selections.

    Returns:
        OrderedDict mapping test index to its score and the original indices/similarities
        of its k_nearest neighbors found in the representative set.
    """

    # --- Prepare features for calculation ---
    # Use only the uncertain subset of the test features
    test_activations = test.activations[indices_test].to(device)
    test_embeds_subset = test.embeds[indices_test].to(device)

    # Use features of the representative training subset
    train_activations = train.activations.to(device)
    train_embeds = train.embeds.to(device) # Already on device from creation if done right

    # Ensure original train indices are on the correct device
    original_train_indices = original_train_indices.to(device)

    # --- Calculate Expected Cosine Similarity ---
    source_B_factor = source_covariance.B_inv.diagonal().to(device) # Ensure B factor is on device

    # Calculate diagonal variance terms
    train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, source_covariance.A_inv.to(device), train_activations)[:,None] * source_B_factor
    test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, source_covariance.A_inv.to(device), test_activations)[:,None] * source_B_factor

    # Calculate expected squared norm
    norm_train_sq = train_embeds**2 + train_diag_cov
    expect_norm_train = norm_train_sq.sum(dim=-1, keepdim=True)
    norm_test_sq = test_embeds_subset**2 + test_diag_cov
    expect_norm_test = norm_test_sq.sum(dim=-1, keepdim=True)

    # Clamp to avoid sqrt(0) or negative values from numerical instability
    expect_norm_train = torch.clamp(expect_norm_train, min=1e-12)
    expect_norm_test = torch.clamp(expect_norm_test, min=1e-12)

    # Normalize expected embeddings for expected cosine similarity calculation
    norm_factor_train = torch.sqrt(expect_norm_train)
    norm_factor_test = torch.sqrt(expect_norm_test)
    embeds_train_normalized = train_embeds / norm_factor_train
    embeds_test_normalized = test_embeds_subset / norm_factor_test

    # Compute expected cosine similarity: [N_test_subset, N_representatives]
    expected_similarity = embeds_test_normalized @ embeds_train_normalized.t()

    # --- Find Top K neighbors ensuring enough unique overall selections ---
    n_representatives = len(train.embeds)
    n_test_subset = len(indices_test)
    total_neighbors_required = k_nearest * n_test_subset

    # Fetch initial neighbors (k_nearest + buffer) for each test sample
    k_fetch = min(k_nearest + buffersize, n_representatives)
    topk = expected_similarity.topk(k_fetch, dim=1)

    # Iteratively increase k_prime until we have enough unique neighbors globally
    k_prime = k_nearest
    first_unique_indices = None # To store the final set of local indices needed
    while True:
        # Get local indices for top k_prime neighbors for all test samples
        # Indices are within the representative set [0, n_representatives-1]
        current_topk_indices_local = topk.indices[:, :k_prime]
        # Flatten to check unique count across all test samples' neighbors
        flat_indices_local = current_topk_indices_local.T.flatten()
        unique_indices_local = torch.unique(flat_indices_local, sorted=False)
        num_unique_found = len(unique_indices_local)

        print(f"K_prime: {k_prime}, Unique indices found: {num_unique_found}, Goal size: {total_neighbors_required}", flush=True)

        # Check if we have enough unique indices OR if we've exhausted the pool
        if num_unique_found >= total_neighbors_required or k_prime >= n_representatives:
            # Need to truncate the flat_indices_local to get exactly total_neighbors_required unique ones
            first_unique_indices = _remove_last_elements_to_keep_n_unique(flat_indices_local, total_neighbors_required)
            unique_indices_final_local = torch.unique(first_unique_indices, sorted=False)
            print(f"Final selection needs {len(unique_indices_final_local)} unique local indices (indices within representative set).")
            break

        # If not enough unique indices, increase k_prime for next iteration
        k_prime += 1
        if k_prime > k_fetch: # Need to fetch more if buffer wasn't enough initially
            print("Buffer size potentially too small, fetching more neighbors...")
            k_fetch = min(k_prime + buffersize, n_representatives)
            topk = expected_similarity.topk(k_fetch, dim=1)


    # --- Map local indices back to original indices and store results ---
    text_idx_to_train_data = OrderedDict()
    # Create a set for fast checking of required unique local indices
    required_unique_local_set = set(unique_indices_final_local.tolist())

    for i, (topk_idx_local, topk_val) in enumerate(zip(topk.indices, topk.values)):
        test_idx = indices_test[i].item() # Original index of the test sample
        test_value = values_test[i].item() # Original score of the test sample

        # Consider top k_prime neighbors found for this test sample
        topk_idx_local_i = topk_idx_local[:k_prime]
        topk_val_i = topk_val[:k_prime]

        keep_ids_original, keep_val = [], []
        count_added = 0
        for idx_local, val in zip(topk_idx_local_i, topk_val_i):
            idx_local_item = idx_local.item()
            # Check if this local index is among the ones globally required
            if idx_local_item in required_unique_local_set:
                # Map local index back to the original training set index
                original_idx = original_train_indices[idx_local_item].item()
                keep_ids_original.append(original_idx)
                keep_val.append(val.item())
                count_added += 1
                # Ensure we don't add more than k_nearest for this specific test sample if k_prime > k_nearest
                if count_added >= k_nearest:
                     break

        # Store original indices and their corresponding similarities
        text_idx_to_train_data[test_idx] = dict(
            score=test_value,
            indices=keep_ids_original, # List of original training set indices
            similarities=keep_val,     # List of corresponding similarity scores
        )

    return text_idx_to_train_data

def find_similar_samples_wasserstein(
    train: EncoderResult,                 # Features of the representative training subset
    test: EncoderResult,                  # Features of the full test set
    indices_test: torch.Tensor,           # Indices of the uncertain test samples
    values_test: torch.Tensor,            # Scores/values of the uncertain test samples
    original_train_indices: torch.Tensor, # Original indices (in full train set) of the representatives
    k_nearest: int,
    source_covariance,
    device: str,
    buffersize=150,                       # Buffer for ensuring unique neighbours
):
    """
    Finds k_nearest neighbors using Squared Diagonal Wasserstein Distance in the representative
    training subset for given uncertain test samples, mapping back to original training indices.

    Args:
        train: EncoderResult for the representative training samples.
        test: EncoderResult for the full test set.
        indices_test: Indices (within 'test') of uncertain samples to find neighbors for.
        values_test: Acquisition scores for the uncertain test samples.
        original_train_indices: Indices (within the original full training set) corresponding
                                to the samples in 'train'.
        k_nearest: Number of nearest neighbors required per test sample.
        source_covariance: Covariance object for source embeddings (needed for distances).
        device: Device to run the computations on.
        buffersize: How many extra neighbors to fetch initially to handle non-unique selections.

    Returns:
        OrderedDict mapping test index to its score and the original indices/similarities
        (negative distances) of its k_nearest neighbors found in the representative set.
    """

    # --- Prepare features for calculation ---
    # Use only the uncertain subset of the test features
    test_activations = test.activations[indices_test].to(device)
    test_embeds = test.embeds[indices_test].to(device) # Renamed to match wdist call

    # Use features of the representative training subset
    train_activations = train.activations.to(device)
    train_embeds = train.embeds.to(device) # Renamed to match wdist call

    # Ensure original train indices are on the correct device
    original_train_indices = original_train_indices.to(device)

    # --- Calculate Squared Diagonal Wasserstein Distance ---
    source_B_factor = source_covariance.B_inv.diagonal().to(device)

    # Calculate diagonal variance terms for the representative train set
    train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, source_covariance.A_inv.to(device), train_activations).unsqueeze(1) * source_B_factor
    train_diag_cov = train_diag_cov.squeeze(1) # Shape [N_representatives, D]

    # Calculate diagonal variance terms for the test subset
    test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, source_covariance.A_inv.to(device), test_activations).unsqueeze(1) * source_B_factor
    test_diag_cov = test_diag_cov.squeeze(1) # Shape [N_test_subset, D]


    # Compute pairwise squared Wasserstein distances: [N_test_subset, N_representatives]
    # Lower distance means more similar
    wasserstein_distances_sq = wdist2(test_embeds, train_embeds, test_diag_cov, train_diag_cov)

    # Convert distances to similarities by negating; higher similarity is better
    similarities = -wasserstein_distances_sq

    # --- Find Top K neighbors ensuring enough unique overall selections ---
    n_representatives = len(train.embeds)
    n_test_subset = len(indices_test)
    total_neighbors_required = k_nearest * n_test_subset

    # Fetch initial neighbors (k_nearest + buffer) based on highest similarity (lowest distance)
    k_fetch = min(k_nearest + buffersize, n_representatives)
    topk = similarities.topk(k_fetch, dim=1)

    # Iteratively increase k_prime until we have enough unique neighbors globally
    k_prime = k_nearest
    first_unique_indices = None # To store the final set of local indices needed
    while True:
        # Get local indices for top k_prime neighbors for all test samples
        current_topk_indices_local = topk.indices[:, :k_prime]
        flat_indices_local = current_topk_indices_local.T.flatten()
        unique_indices_local = torch.unique(flat_indices_local, sorted=False)
        num_unique_found = len(unique_indices_local)

        print(f"K_prime: {k_prime}, Unique indices found: {num_unique_found}, Goal size: {total_neighbors_required}", flush=True)

        if num_unique_found >= total_neighbors_required or k_prime >= n_representatives:
            first_unique_indices = _remove_last_elements_to_keep_n_unique(flat_indices_local, total_neighbors_required)
            unique_indices_final_local = torch.unique(first_unique_indices, sorted=False)
            print(f"Final selection needs {len(unique_indices_final_local)} unique local indices (indices within representative set).")
            break

        k_prime += 1
        if k_prime > k_fetch:
             print("Buffer size potentially too small, fetching more neighbors...")
             k_fetch = min(k_prime + buffersize, n_representatives)
             topk = similarities.topk(k_fetch, dim=1)


    # --- Map local indices back to original indices and store results ---
    text_idx_to_train_data = OrderedDict()
    required_unique_local_set = set(unique_indices_final_local.tolist())

    for i, (topk_idx_local, topk_val) in enumerate(zip(topk.indices, topk.values)):
        test_idx = indices_test[i].item()
        test_value = values_test[i].item()

        topk_idx_local_i = topk_idx_local[:k_prime]
        topk_val_i = topk_val[:k_prime] # These are negative distances

        keep_ids_original, keep_val = [], []
        count_added = 0
        for idx_local, val in zip(topk_idx_local_i, topk_val_i):
             idx_local_item = idx_local.item()
             if idx_local_item in required_unique_local_set:
                 original_idx = original_train_indices[idx_local_item].item()
                 keep_ids_original.append(original_idx)
                 keep_val.append(val.item()) # Store similarity (neg distance)
                 count_added += 1
                 if count_added >= k_nearest:
                     break

        text_idx_to_train_data[test_idx] = dict(
            score=test_value,
            indices=keep_ids_original,
            similarities=keep_val, # Storing similarities (negative distances)
        )

    return text_idx_to_train_data