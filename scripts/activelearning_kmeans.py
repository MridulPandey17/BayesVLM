# activelearning.py

import argparse
import copy
import json
from typing import Literal
from collections import OrderedDict
from pathlib import Path

import torch
import torch.utils.data
import wandb
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_calibration_error,
)
from tqdm import tqdm

# Clustering related imports
from sklearn.cluster import KMeans # Or MiniBatchKMeans for large datasets
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

from bayesvlm.vlm import CLIP, EncoderResult
from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.precompute import precompute_image_features, precompute_text_features, make_predictions
from bayesvlm.epig import select_epig_online
from bayesvlm.knn import (
    extract_test_train_indices,
    find_similar_samples_cosine,
    find_similar_samples_wasserstein,
)
from bayesvlm.selection import select_random, select_topk
from bayesvlm.hessians import compute_covariances, load_hessians, optimize_prior_precision
from bayesvlm.utils import get_model_type_and_size, get_image_size, get_transform, load_model


def evaluate(
    projection: torch.nn.Module,
    text_outputs: EncoderResult,
    clip: CLIP,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
):
    """Evaluates the model on a given data loader."""
    clip = clip.eval().to(device)
    projection = projection.eval().to(device)
    text_outputs = text_outputs.to(device)

    all_logits_mean = []
    all_logits_var = []
    all_labels = []
    total_loss = 0.0
    with torch.no_grad():
        for activations, residuals, lbls in loader:
            activations, residuals, lbls = activations.to(device), residuals.to(device), lbls.to(device)
            image_embeds = projection(activations) + residuals
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)

            logits = clip(image_outputs, text_outputs) # This should return a Distribution object if CLIP is Bayesian enabled
            
            # Assuming logits object has mean and var attributes (adjust if needed)
            current_logits_mean = logits.mean if hasattr(logits, 'mean') else logits
            current_logits_var = logits.var if hasattr(logits, 'var') else torch.zeros_like(current_logits_mean) # Placeholder variance if not Bayesian

            all_logits_mean.append(current_logits_mean.cpu())
            all_logits_var.append(current_logits_var.cpu())
            all_labels.append(lbls.cpu())

            # Calculate loss using mean predictions for evaluation consistency
            loss = torch.nn.functional.cross_entropy(current_logits_mean, lbls, reduction='sum')
            total_loss += loss.item()

    all_logits_mean = torch.cat(all_logits_mean, dim=0)
    all_logits_var = torch.cat(all_logits_var, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Ensure tensors are on CPU before metric calculation if needed by torchmetrics
    all_logits_mean_cpu = all_logits_mean.cpu()
    all_labels_cpu = all_labels.cpu()

    acc = multiclass_accuracy(all_logits_mean_cpu, all_labels_cpu, num_classes=num_classes, average='micro')
    acc_weighted = multiclass_accuracy(all_logits_mean_cpu, all_labels_cpu, num_classes=num_classes, average='weighted')
    ece = multiclass_calibration_error(all_logits_mean_cpu, all_labels_cpu, num_classes=num_classes)

    return dict(
        accuracy=acc.item(),
        accuracy_weighted=acc_weighted.item(),
        ece=ece.item(),
        loss=total_loss / len(loader.dataset),
    )

def cluster_and_select_representatives(
    train_outputs: EncoderResult,
    k_clusters: int,
    device: str,
    seed: int = 0,
    use_minibatch: bool = False,
    batch_size_kmeans: int = 1024
) -> tuple[torch.Tensor, EncoderResult]:
    """
    Performs K-Means (or MiniBatchKMeans) clustering on training embeddings
    and selects representative samples closest to the centroids.

    Args:
        train_outputs: EncoderResult containing embeddings and activations for the full training set.
        k_clusters: The number of clusters (and representatives) to find.
        device: The device to use for intermediate torch operations.
        seed: Random seed for KMeans.
        use_minibatch: Whether to use MiniBatchKMeans for large datasets.
        batch_size_kmeans: Batch size if using MiniBatchKMeans.


    Returns:
        A tuple containing:
        - representative_indices: Tensor of indices (within the original train_outputs)
                                 of the selected representative samples, on the specified device.
        - representative_outputs: EncoderResult containing features ONLY for the representatives,
                                  with tensors on the specified device.
    """
    print(f"    - Clustering training data into {k_clusters} clusters...")
    # KMeans works best on CPU with numpy
    train_embeds_np = train_outputs.embeds.cpu().numpy()

    if use_minibatch:
         print(f"    - Using MiniBatchKMeans with batch size {batch_size_kmeans}.")
         kmeans = MiniBatchKMeans(
             n_clusters=k_clusters,
             random_state=seed,
             batch_size=batch_size_kmeans,
             n_init=10, # Increased n_init
             max_iter=300 # Default max_iter
         )
    else:
        print(f"    - Using standard KMeans.")
        kmeans = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10, verbose=0) # n_init='auto' in newer sklearn

    kmeans.fit(train_embeds_np)
    centroids = kmeans.cluster_centers_

    print(f"    - Finding training samples closest to {k_clusters} centroids...")
    # Find the index of the training sample closest to each centroid
    # pairwise_distances_argmin_min computes distances between centroids (X) and samples (Y)
    # It returns the indices in Y that are closest to each point in X
    closest_indices_np, _ = pairwise_distances_argmin_min(centroids, train_embeds_np)
    representative_indices = torch.tensor(closest_indices_np, dtype=torch.long)

    # Ensure unique indices if multiple centroids map to the same closest point (can happen)
    representative_indices = torch.unique(representative_indices)
    num_representatives = len(representative_indices)
    print(f"    - Selected {num_representatives} unique representative training samples.")
    if num_representatives < k_clusters:
        print(f"    - Warning: Found fewer unique representatives ({num_representatives}) than requested clusters ({k_clusters}).")


    # Create a new EncoderResult containing only the representative features, ensuring they are on the target device
    representative_outputs = EncoderResult(
        embeds=train_outputs.embeds[representative_indices].clone().to(device),
        activations=train_outputs.activations[representative_indices].clone().to(device),
        residuals=train_outputs.residuals[representative_indices].clone().to(device),
    )

    return representative_indices.to(device), representative_outputs

def run_knn(
    embeds_train: EncoderResult,           # Representative features
    embeds_test: EncoderResult,            # Full test set features
    indices_test: torch.Tensor,            # Indices of uncertain test samples
    values_test: torch.Tensor,             # Scores of uncertain test samples
    original_train_indices: torch.Tensor,  # Original indices of representatives
    k_nearest: int,
    device: str,
    source_covariance,
    method: str,
    proj_has_bias=False,                   # Check if needed - likely not changing
    buffersize=150
):
    """ Wrapper to call the appropriate k-NN finding function. """
    # Optional bias handling (check if still relevant - probably keep as is)
    if proj_has_bias:
        # Important: Clone only if modifying inplace is necessary, otherwise just pass through.
        # Let's assume bias handling is done internally or not needed for knn.py funcs for now.
        print("Warning: proj_has_bias=True in run_knn not fully implemented with representative logic.")
        # Pass through unmodified representative features for now.
        pass


    # Pass original_train_indices to the specific implementation
    if method == 'cosine':
        return find_similar_samples_cosine(
            train=embeds_train,             # Representative features
            test=embeds_test,               # Full test features
            indices_test=indices_test,      # Uncertain test indices
            values_test=values_test,        # Uncertain test scores
            original_train_indices=original_train_indices, # Mapping indices
            k_nearest=k_nearest,
            source_covariance=source_covariance,
            device=device,
            buffersize=buffersize
        )
    elif method == 'wasserstein':
        return find_similar_samples_wasserstein(
            train=embeds_train,             # Representative features
            test=embeds_test,               # Full test features
            indices_test=indices_test,      # Uncertain test indices
            values_test=values_test,        # Uncertain test scores
            original_train_indices=original_train_indices, # Mapping indices
            k_nearest=k_nearest,
            source_covariance=source_covariance,
            device=device,
            buffersize=buffersize
        )
    else:
        raise ValueError(f"Unknown knn method {method}")


def finetune(
    img_projection: torch.nn.Module,
    txt_projection: torch.nn.Module,
    clip: CLIP,
    image_features_train: EncoderResult,  # This will be the selected subset
    labels_train: torch.Tensor,
    image_features_val: EncoderResult,
    labels_val: torch.Tensor,
    image_features_test: EncoderResult,
    labels_test: torch.Tensor,
    text_features: EncoderResult,         # Text features for all classes
    lr: float,
    wd: float,
    epochs: int,
    batch_size: int,
    device: str,
    finetune_dir: Path,
    selection: str,
    num_classes: int,
    k_nearest: int,
    subset_size: int,
    dataset: str,
    hessian_scale: float,
    project_name: str,
    epig_lr: float,
    epig_hessian_update_scale: float,
    epig_mc_samples: int = 100,           # Make sure this matches definition if used elsewhere
    knn_method: str = 'wasserstein',
    kmeans_clusters: int | None = None, # Added for logging
):
    """ Fine-tunes the image projection layer on a selected subset of training data. """
    if not wandb.run:
         print("Initializing WandB for finetuning run...")
         wandb.init(project=project_name, dir=str(finetune_dir), reinit=True)
    else:
         print("WandB already initialized.")


    # Update WandB config - ensure all relevant params are logged
    config_updates = {
        'finetune_lr': lr,
        'finetune_wd': wd,
        'finetune_epochs': epochs,
        'finetune_batch_size': batch_size,
        'selection_method': selection, # Renamed for clarity
        'subset_size': subset_size,
        'knn_k_nearest': k_nearest, # Renamed for clarity
        'dataset': dataset,
        'hessian_scale': hessian_scale,
        'epig_lr': epig_lr,
        'epig_hessian_update_scale': epig_hessian_update_scale,
        'epig_mc_samples': epig_mc_samples,
        'knn_method': knn_method,
        'kmeans_clusters': kmeans_clusters, # Log k-means setting
    }
    wandb.config.update(config_updates, allow_val_change=True)


    # Set run name if not already set (might be set by the main loop)
    if wandb.run.name is None or 'Unnamed run' in wandb.run.name :
        run_name_parts = [finetune_dir.parent.name, finetune_dir.name]
        wandb.run.name = '/'.join(run_name_parts)
        wandb.run.save()
        print(f"WandB Run Name set to: {wandb.run.name}")

    # --- Model Preparation ---
    clip = clip.eval().to(device) # Ensure base CLIP is eval mode
    # Freeze parts we are not training
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None:
        clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None:
         clip.logit_bias.data.requires_grad = False


    txt_projection = txt_projection.eval().to(device)
    for param in txt_projection.parameters():
        param.requires_grad = False

    # Ensure image projection is trainable
    img_projection = img_projection.train().to(device)
    params_to_train = []
    for param in img_projection.parameters():
        param.requires_grad = True
        params_to_train.append(param)
        # print(f"Training param: {name}") # Debug

    # --- Dataset Preparation ---
    text_features = text_features.to(device) # Ensure text features are on device

    # Use the provided selected subset for training
    train_ds = torch.utils.data.TensorDataset(image_features_train.activations.cpu(), image_features_train.residuals.cpu(), labels_train.cpu())
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1, # Keep low for small datasets/debugging
        pin_memory=True if device != 'cpu' else False
    )

    val_ds = torch.utils.data.TensorDataset(image_features_val.activations.cpu(), image_features_val.residuals.cpu(), labels_val.cpu())
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size, # Use same batch size for consistency? Or larger for speed?
        shuffle=False,
        num_workers=1,
        pin_memory=True if device != 'cpu' else False
    )

    test_ds = torch.utils.data.TensorDataset(image_features_test.activations.cpu(), image_features_test.residuals.cpu(), labels_test.cpu())
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True if device != 'cpu' else False
    )

    # --- Optimizer ---
    if not params_to_train:
        print("Warning: No parameters found requiring grad for the optimizer.")
        # Handle this case - maybe return original projection?
        return img_projection # Or raise error

    optimizer = torch.optim.AdamW(
        params_to_train,
        lr=lr,
        weight_decay=wd,
    )
    # Scheduler (optional, can add ReduceLROnPlateau or CosineAnnealingLR)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


    # --- Initial Evaluation ---
    print("Evaluating initial model state...")
    initial_train_metrics = evaluate(img_projection, text_features, clip, train_loader, num_classes=num_classes, device=device)
    initial_val_metrics = evaluate(img_projection, text_features, clip, val_loader, num_classes=num_classes, device=device)
    initial_test_metrics = evaluate(img_projection, text_features, clip, test_loader, num_classes=num_classes, device=device)

    wandb.log({f'train/{k}': v for k, v in initial_train_metrics.items()}, step=0)
    wandb.log({f'val/{k}': v for k, v in initial_val_metrics.items()}, step=0)
    wandb.log({f'test/{k}': v for k, v in initial_test_metrics.items()}, step=0)


    # --- Training Loop ---
    best_val_loss = initial_val_metrics.get('loss', float('inf'))
    best_test_metrics = initial_test_metrics
    best_val_metrics = initial_val_metrics
    best_projection_state_dict = copy.deepcopy(img_projection.state_dict())
    patience_counter = 0
    patience = 15 # Stop if validation loss doesn't improve for X epochs

    print("Starting fine-tuning loop...")
    for epoch in range(epochs):
        img_projection.train() # Set model to training mode
        epoch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for activations, residuals, lbls in progress_bar:
            activations, residuals, lbls = activations.to(device), residuals.to(device), lbls.to(device)
            optimizer.zero_grad()

            image_embeds = img_projection(activations) + residuals
            # No need to compute text embeds every step if projection is frozen
            # text_embeds = txt_projection(text_features.activations) # If needed by clip call

            # Assume clip call handles recombination or uses precomputed text embeds
            logits_out = clip(image_embeds, text_features.embeds) # Pass embeds directly if clip handles it
            # If clip expects EncoderResult:
            # image_outputs_batch = EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)
            # logits_out = clip(image_outputs_batch, text_features)


            # Extract mean for loss calculation
            logits_mean = logits_out.mean if hasattr(logits_out, 'mean') else logits_out

            loss = torch.nn.functional.cross_entropy(logits_mean, lbls)

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch + 1}/{epochs}, Average Train Loss: {avg_epoch_loss:.4f}")

        # --- Evaluation after epoch ---
        img_projection.eval() # Set model to evaluation mode
        current_train_metrics = evaluate(img_projection, text_features, clip, train_loader, num_classes=num_classes, device=device)
        current_val_metrics = evaluate(img_projection, text_features, clip, val_loader, num_classes=num_classes, device=device)
        current_test_metrics = evaluate(img_projection, text_features, clip, test_loader, num_classes=num_classes, device=device)

        # Update scheduler based on validation loss (if using scheduler)
        # scheduler.step(current_val_metrics['loss'])

        # --- Checkpointing and Early Stopping ---
        current_val_loss = current_val_metrics['loss']
        if current_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} -> {current_val_loss:.4f}). Saving model.")
            best_val_loss = current_val_loss
            best_val_metrics = current_val_metrics
            best_test_metrics = current_test_metrics
            best_projection_state_dict = copy.deepcopy(img_projection.state_dict())
            patience_counter = 0 # Reset patience
            # Save the best model state immediately
            torch.save(best_projection_state_dict, finetune_dir / 'best_img_projection.pt')
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")


        # --- Logging ---
        wandb.log({f'train/loss_epoch': avg_epoch_loss}, step=epoch + 1)
        wandb.log({f'train/{k}': v for k, v in current_train_metrics.items()}, step=epoch + 1)
        wandb.log({f'val/{k}': v for k, v in current_val_metrics.items()}, step=epoch + 1)
        wandb.log({f'test/{k}': v for k, v in current_test_metrics.items()}, step=epoch + 1)
        wandb.log({'val/best_loss': best_val_loss}, step=epoch+1) # Log best loss achieved so far

        if best_test_metrics is not None: # Should always be true after first eval
            wandb.log({f'best_model/val_{k}': v for k, v in best_val_metrics.items()}, step=epoch + 1)
            wandb.log({f'best_model/test_{k}': v for k, v in best_test_metrics.items()}, step=epoch + 1)


        # --- Early Stopping ---
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("Finished fine-tuning.")
    # Load the best performing state dict back into the model before returning
    img_projection.load_state_dict(best_projection_state_dict)
    return img_projection


def main(
    model_str: str,
    dataset: str,
    hessian_dir: str,
    experiment_dir: str,
    project_name: str,
    hessian_scale: float,
    subset_size: int,
    kmeans_clusters: int | None, # Added K-Means argument
    use_minibatch_kmeans: bool,  # Added flag for MiniBatchKMeans
    kmeans_batch_size: int,      # Added batch size for MiniBatchKMeans
    predictions_batch_size: int = 256,
    precompute_batch_size: int = 256,
    precompute_num_workers: int = 8,
    finetune_lr: float = 1e-5,
    finetune_wd: float = 5e-2,
    finetune_epochs: int = 100,
    finetune_batch_size: int = 30,
    only_deterministic_strategies: bool = False,
    only_random_strategies: bool = False,
    only_epig: bool = False,
    without_epig: bool = False,
    epig_lr: float = 1e-4,
    epig_hessian_update_scale: float = 10.0,
    epig_num_samples: int = 100,
    k_nearest: int = 1,
    knn_method: Literal['cosine', 'wasserstein'] = 'wasserstein',
    knn_buffersize: int = 150, # Add buffer size argument
    device: str = 'cuda',
):
    run_dir = Path(experiment_dir) / dataset
    if not run_dir.exists():
        print(f"Creating run directory {run_dir}")
        run_dir.mkdir(parents=True)

    # --- Setup Model and Data ---
    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    factory = DataModuleFactory(
        batch_size=precompute_batch_size,
        num_workers=precompute_num_workers,
        shuffle_train=False, # Important: Keep false for consistent indexing
        train_transform=transform,
        test_transform=transform,
    )
    dm = factory.create(dataset)
    dm.setup()

    # Set K-Means clusters default if not provided
    if kmeans_clusters is None:
        # Heuristic: Needs enough diversity but shouldn't be excessively large
        kmeans_clusters = min(len(dm.train_dataloader().dataset) // 2, max(50, subset_size * 10))
        print(f"Setting --kmeans_clusters automatically to {kmeans_clusters}")


    # --- Load/Compute Features ---
    print("[0] Loading base models...")
    image_encoder, text_encoder, clip = load_model(model_str, device=device)
    image_encoder.freeze_all_layers() # Keep frozen
    text_encoder.freeze_all_layers() # Keep frozen
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None:
        clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None:
        clip.logit_bias.data.requires_grad = False

    print("[1] Precomputing features ...")
    # Define cache dirs based on base model to avoid recomputing if model doesn't change
    base_cache_dir = run_dir / f"features_{model_str.replace('/', '_')}" # Unique name
    image_outputs_train, image_class_ids_train, image_ids_train = precompute_image_features(
        image_encoder=image_encoder, loader=dm.train_dataloader(),
        cache_dir=base_cache_dir / 'train', save_predictions=True, device=device, # Ensure device used
    )
    image_outputs_val, image_class_ids_val, image_ids_val = precompute_image_features(
        image_encoder=image_encoder, loader=dm.val_dataloader(),
        cache_dir=base_cache_dir / 'val', save_predictions=True, device=device,
    )
    image_outputs_test, image_class_ids_test, image_ids_test = precompute_image_features(
        image_encoder=image_encoder, loader=dm.test_dataloader(),
        cache_dir=base_cache_dir / 'test', save_predictions=True, device=device,
    )
    label_outputs = precompute_text_features(
        text_encoder=text_encoder, class_prompts=dm.class_prompts,
        batch_size=precompute_batch_size, cache_dir=base_cache_dir, save_predictions=True, device=device
    )

    # --- Load Hessians and Compute Covariances ---
    print("[1.5] Loading Hessians and computing covariances...")
    A_img, B_img = load_hessians(la_dir=hessian_dir, tag='img', return_info=False)
    A_txt, B_txt, info = load_hessians(la_dir=hessian_dir, tag='txt', return_info=True)

    # Optimize prior precision (can be skipped if using fixed values)
    lambda_img = optimize_prior_precision(
        projection=image_encoder.vision_projection, A=A_img, B=B_img,
        lmbda_init=info.get('lambda_img', 1.0), n=hessian_scale, device=device, retain_graph=False,
    ).item()
    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection, A=A_txt, B=B_txt,
        lmbda_init=info.get('lambda_txt', 1.0), n=hessian_scale, device=device, retain_graph=False,
    ).item()
    print(f"Optimized lambda_img: {lambda_img:.4f}, lambda_txt: {lambda_txt:.4f}")

    covar_info = dict(lambda_img=lambda_img, lambda_txt=lambda_txt, n_img=hessian_scale, n_txt=hessian_scale)
    cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, covar_info)
    clip.set_covariances(source_covariance=cov_img, target_covariance=cov_txt)


    # --- Make Base Predictions ---
    print("[2] Making base predictions ...")
    prob_logits_train = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, save_predictions=False, map_estimate=False)
    prob_logits_train_map = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, save_predictions=False, map_estimate=True)
    prob_logits_val = make_predictions(clip=clip, image_outputs=image_outputs_val, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, save_predictions=False, map_estimate=False) # Val predictions needed? Maybe not.
    prob_logits_test = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, save_predictions=False, map_estimate=False)
    prob_logits_test_map = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, save_predictions=False, map_estimate=True)

    # --- K-Means Clustering ---
    print("[2.5] Clustering training set to find representatives...")
    representative_train_indices, representative_train_outputs = cluster_and_select_representatives(
        train_outputs=image_outputs_train,
        k_clusters=kmeans_clusters,
        device=device, # Representatives kept on device
        seed=0, # Use a fixed seed
        use_minibatch=use_minibatch_kmeans,
        batch_size_kmeans=kmeans_batch_size
    )

    # Define directory for this specific run configuration
    # Include K-Means cluster count in the path name for clarity
    path_suffix = f'subset_{subset_size}_k_{k_nearest}_n_{hessian_scale}_kmeans_{kmeans_clusters}_epig_lr_{epig_lr}_epig_update_{epig_hessian_update_scale}_knn_{knn_method}'
    subset_dir = run_dir / path_suffix
    if not subset_dir.exists():
        print(f"Creating subset directory: {subset_dir}")
        subset_dir.mkdir(parents=True)
    else:
         print(f"Using existing subset directory: {subset_dir}")

    print("[3] Creating training subsets ...")
    json_path = subset_dir / 'subset_indices_train.json'
    if json_path.exists():
        print(f"Loading existing subset indices from {json_path}")
        with open(json_path) as f:
            subset_indices_train = json.load(f)
            # Convert lists back to OrderedDict if needed (might lose order from json)
            subset_indices_train = OrderedDict(subset_indices_train)
    else:
        print(f"Generating new subset indices (will save to {json_path})")
        subset_indices_train = OrderedDict()

    # --- Define Subset Generation Logic ---
    if not only_random_strategies and not only_epig:
        # Aleatoric entropy (MAP predictions) + K-Means/KNN Targeting
        strategy_key = 'entropy_map_kmeans_knn'
        print(f"    - Generating subset for: {strategy_key}...", flush=True)
        if strategy_key not in subset_indices_train:
            indices_entropy_alea_test_map, values_entropy_alea_test_map = select_topk(prob_logits_test_map, k=subset_size, variant='entropy', entropy_variant='map_alea', return_values=True)
            indices_entropy_alea_support = run_knn(
                embeds_train=representative_train_outputs, # Search within representatives
                embeds_test=image_outputs_test,
                indices_test = indices_entropy_alea_test_map,
                values_test = values_entropy_alea_test_map,
                original_train_indices=representative_train_indices, # Map back
                k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method,
                proj_has_bias=clip.source_projection_has_bias, buffersize=knn_buffersize,
            )
            subset_indices_train[strategy_key] = indices_entropy_alea_support
            print(f"    - Done generating subset for: {strategy_key}")

        # Aleatoric entropy (MAP predictions) on FULL Trainset (No Targeting)
        strategy_key = 'entropy_map_trainset_direct'
        print(f"    - Generating subset for: {strategy_key}...", flush=True)
        if strategy_key not in subset_indices_train:
            indices_entropy_alea_train_map, values_entropy_alea_train_map = select_topk(prob_logits_train_map, k=subset_size, variant='entropy', entropy_variant='map_alea', return_values=True)
            subset_indices_train[strategy_key] = { 0: dict( score=0.0, indices=indices_entropy_alea_train_map.tolist(), similarities=values_entropy_alea_train_map.tolist()) } # Hacky structure match
            print(f"    - Done generating subset for: {strategy_key}")


        # BALD (Bayesian uncertainty) + K-Means/KNN Targeting
        strategy_key = f'bald_test_kmeans_knn'
        print(f"    - Generating subset for: {strategy_key}...", flush=True)
        if strategy_key not in subset_indices_train:
            indices_entropy_exp_mutual_info_test, values_entropy_exp_mutual_info_test = select_topk(prob_logits_test, k=subset_size, variant='exp_mutual_info', return_values=True, seed=0)
            indices_entropy_exp_mutual_info_support = run_knn(
                embeds_train=representative_train_outputs, # Search within representatives
                embeds_test=image_outputs_test,
                indices_test = indices_entropy_exp_mutual_info_test,
                values_test = values_entropy_exp_mutual_info_test,
                original_train_indices=representative_train_indices, # Map back
                k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method,
                proj_has_bias=clip.source_projection_has_bias, buffersize=knn_buffersize,
            )
            subset_indices_train[strategy_key] = indices_entropy_exp_mutual_info_support
            print(f"    - Done generating subset for: {strategy_key}")

    if not only_random_strategies and not without_epig:
        # EPIG + K-Means/KNN Subsampling Targeting
        # Note: Adjust EPIG's internal subsampling if needed, or assume 'knn' searches representatives now.
        # Let's assume EPIG still does its own thing or uses the *full* pool for its search.
        # This needs careful thought about how EPIG integrates with pre-filtering.
        # Option 1: EPIG selects from full pool, THEN we do Kmeans/KNN (less efficient).
        # Option 2: EPIG's pool_subsampling uses representatives (might change EPIG behavior).
        # Let's add EPIG but maybe *without* the kmeans pre-filter for now, needs careful integration.
        strategy_key = f'epig_direct' # Indicates EPIG runs on its potentially subsampled full pool view
        print(f"    - Generating subset for: {strategy_key} (Note: K-Means pre-filter NOT applied here)...", flush=True)
        if strategy_key not in subset_indices_train:
            # Choose pool subsampling for EPIG: None, random, or maybe 'knn' (if EPIG's knn uses full pool)
            epig_pool_subsampling = 'knn_wasserstein' if knn_method == 'wasserstein' else 'knn_cosine' # Default EPIG behavior from original code

            indices_epig, epig_scores = select_epig_online(
                label_features=label_outputs, pool_features=image_outputs_train, target_features=image_outputs_test,
                pool_class_ids=image_class_ids_train, image_projection=image_encoder.vision_projection, clip=clip,
                A_img=A_img, B_img=B_img, A_txt=A_txt, B_txt=B_txt, cov_info=covar_info, budget=subset_size,
                lr=epig_lr, hessian_update_scale=epig_hessian_update_scale, device=device, num_samples=epig_num_samples,
                seed=0, pool_max_size=40_000, target_max_size=20_000, # Consider adjusting these
                pool_subsampling = epig_pool_subsampling,
                proj_has_bias=clip.source_projection_has_bias,
            )
            subset_indices_train[strategy_key] = { 0: dict( score=0.0, indices=indices_epig, similarities=epig_scores)}
            print(f"    - Done generating subset for: {strategy_key}")


    if not only_deterministic_strategies and not only_epig:
        # Random Selection on Test + K-Means/KNN Targeting
        for i in range(5): # Generate multiple random runs
             strategy_key = f'random_on_test_kmeans_knn_{i}'
             print(f"    - Generating subset for: {strategy_key}...", flush=True)
             if strategy_key not in subset_indices_train:
                 indices_random_test = select_random(prob_logits_test, k=subset_size, seed=i)
                 # Use float for consistency if values_test expected float later
                 values_random_test = torch.ones_like(indices_random_test, dtype=torch.float32)
                 indices_random_support = run_knn(
                     embeds_train=representative_train_outputs, # Search within representatives
                     embeds_test=image_outputs_test,
                     indices_test=indices_random_test,
                     values_test=values_random_test,
                     original_train_indices=representative_train_indices, # Map back
                     k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method,
                     proj_has_bias=clip.source_projection_has_bias, buffersize=knn_buffersize,
                 )
                 subset_indices_train[strategy_key] = indices_random_support
                 print(f"    - Done generating subset for: {strategy_key}")

        # Random Selection directly on Full Trainset (No Targeting)
        for i in range(5):
            strategy_key = f'random_on_train_direct_{i}'
            print(f"    - Generating subset for: {strategy_key}...", flush=True)
            if strategy_key not in subset_indices_train:
                # Select K*N for compatibility, although we only need N unique.
                # select_random should handle pool size limits correctly.
                needed_samples = k_nearest * subset_size # Match old logic goal, though uniqueness handled later
                indices_random_trivial_support = select_random(prob_logits_train, k=needed_samples, seed=i)

                # Ensure we have subset_size *unique* indices if possible
                unique_indices = torch.unique(indices_random_trivial_support)
                if len(unique_indices) > subset_size:
                    indices_random_trivial_support = unique_indices[:subset_size]
                elif len(unique_indices) < subset_size:
                    print(f"Warning: random_on_train found only {len(unique_indices)} unique samples, requested {subset_size}. Using available.")
                    indices_random_trivial_support = unique_indices


                subset_indices_train[strategy_key] = {0: dict( score=0.0, indices=indices_random_trivial_support.tolist(), similarities=[1.0] * len(indices_random_trivial_support))}
                print(f"    - Done generating subset for: {strategy_key}")

    # Save the generated dictionary of subsets
    try:
        with open(json_path, 'w') as f:
            # Convert OrderedDict to dict for simple JSON serialization
            json.dump(dict(subset_indices_train), f, indent=4)
        print(f"Saved subset indices to {json_path}")
    except Exception as e:
        print(f"Error saving subset indices to {json_path}: {e}")


    # --- Fine-tuning Loop ---
    print("[4] Fine-tuning based on generated training subsets ...")
    # Initialize WandB once before the loop if running multiple finetunes in one script execution
    # Make sure wandb finishes run before starting next if called inside loop like before.
    # For now, assuming one finetune per script call, handled inside finetune function.

    for subset_key, indices_dict in subset_indices_train.items():
        print(f"--- Starting Finetune for Subset: {subset_key} ---")

        # Extract the actual training indices needed for this subset
        # Logic differs slightly: direct selection vs. k-NN mapping
        if 'direct' in subset_key or 'epig' in subset_key: # Directly selected train indices
            final_train_indices = torch.tensor(indices_dict[0]['indices'], dtype=torch.long)
        else: # Indices derived from k-NN mapping structure
             extracted = extract_test_train_indices(indices_dict)
             final_train_indices = torch.tensor(extracted['train'], dtype=torch.long)

        # Ensure unique indices and cap at subset_size
        final_train_indices = torch.unique(final_train_indices)
        if len(final_train_indices) > subset_size:
             print(f"Warning: Subset {subset_key} initially had {len(final_train_indices)} unique indices. Capping at {subset_size}.")
             # Randomly sample subset_size from the unique indices for fairness? Or just take first? Let's take first for simplicity.
             final_train_indices = final_train_indices[:subset_size]

        print(f"    - Using {len(final_train_indices)} unique training samples for subset {subset_key}.")

        # Create subset features
        masked_image_features = EncoderResult(
            embeds=image_outputs_train.embeds[final_train_indices],
            activations=image_outputs_train.activations[final_train_indices],
            residuals=image_outputs_train.residuals[final_train_indices]
        )
        masked_class_ids = image_class_ids_train[final_train_indices]
        # masked_image_ids = image_ids_train[final_train_indices] # Keep if needed

        # Define directory for this specific fine-tuning run
        finetune_dir = subset_dir / subset_key
        finetune_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = finetune_dir / 'best_img_projection.pt' # Save best model

        # --- Start Fine-tuning Run ---
        # Make fresh copies of projections for each run
        img_projection_ft = copy.deepcopy(image_encoder.vision_projection)
        txt_projection_ft = copy.deepcopy(text_encoder.text_projection)
        # Ensure base CLIP is also copied if its state might change (unlikely if frozen)
        clip_ft = copy.deepcopy(clip)
        # Set covariances again on the copied clip object
        clip_ft.set_covariances(source_covariance=cov_img, target_covariance=cov_txt)


        # Extract selection method base name for logging
        selection_base_name = '_'.join(subset_key.split('_')[:-1]) if subset_key.split('_')[-1].isdigit() else subset_key

        # Make sure wandb starts a new run for each subset if not running in parallel
        if wandb.run:
            print("Finishing previous WandB run...")
            wandb.finish()

        print(f"    - Starting finetune process for {subset_key}...")
        best_projection = finetune(
            img_projection=img_projection_ft,
            txt_projection=txt_projection_ft,
            clip=clip_ft, # Use copied clip
            image_features_train=masked_image_features,
            labels_train=masked_class_ids,
            image_features_val=image_outputs_val,
            labels_val=image_class_ids_val,
            image_features_test=image_outputs_test,
            labels_test=image_class_ids_test,
            text_features=label_outputs,
            lr=finetune_lr, wd=finetune_wd, epochs=finetune_epochs, batch_size=finetune_batch_size,
            device=device, finetune_dir=finetune_dir, selection=selection_base_name,
            num_classes=len(dm.class_prompts), k_nearest=k_nearest, subset_size=len(final_train_indices), # Log actual size used
            project_name=project_name, dataset=dataset, hessian_scale=hessian_scale,
            epig_lr=epig_lr, epig_hessian_update_scale=epig_hessian_update_scale,
            epig_mc_samples=epig_num_samples, knn_method=knn_method,
            kmeans_clusters=kmeans_clusters # Pass for logging
        )

        # Save the state dict of the best projection returned by finetune
        # Note: finetune already saves 'best_img_projection.pt' based on validation loss.
        # We could skip saving here unless we want the final epoch state regardless of validation.
        # torch.save(best_projection.state_dict(), checkpoint_path.with_name('final_img_projection.pt'))
        print(f"    - Fine-tuning finished for {subset_key}. Best model saved to {finetune_dir / 'best_img_projection.pt'}")

        # Finish wandb run for this subset explicitly if initialized in finetune
        if wandb.run:
             wandb.finish()


if __name__ == '__main__':
    # Prevents potential issues with multiprocessing in PyTorch >= 1.9
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or OS doesn't support 'spawn'.")

    # For reproducibility (optional)
    # torch.manual_seed(42)
    # np.random.seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="Active Fine-tuning with K-Means pre-selection.")
    parser.add_argument('--model', type=str, default='clip-base', help="CLIP model string (e.g., 'clip-base', 'clip-large')")
    parser.add_argument('--dataset', type=str, default='homeoffice-da-clipart', help="Dataset identifier (e.g., 'homeoffice-da-clipart', 'imagenet-r')")
    parser.add_argument('--hessian_dir', type=str, required=True, help="Directory containing precomputed Hessian factors")

    parser.add_argument('--experiment_dir', type=str, default='experiments/active-finetuning-kmeans', help="Base directory for experiment outputs")
    parser.add_argument('--project_name', type=str, default='active-finetuning-kmeans', help="WandB project name")

    # Experiment parameters
    parser.add_argument('--subset_size', type=int, default=50, help="Number of samples in the final fine-tuning subset")
    parser.add_argument('--hessian_scale', type=float, default=10, help="Scaling factor 'n' (pseudo-data count) for covariance calculation")
    parser.add_argument('--kmeans_clusters', type=int, default=None, help='Number of clusters for KMeans pre-selection. Default: heuristic based on subset_size')
    parser.add_argument('--use_minibatch_kmeans', action='store_true', help='Use MiniBatchKMeans instead of standard KMeans')
    parser.add_argument('--kmeans_batch_size', type=int, default=1024, help='Batch size for MiniBatchKMeans')


    # Precompute/Prediction parameters
    parser.add_argument('--predictions_batch_size', type=int, default=256)
    parser.add_argument('--precompute_batch_size', type=int, default=256)
    parser.add_argument('--precompute_num_workers', type=int, default=4, help="Number of workers for data loading during precomputation") # Reduced default

    # Fine-tuning parameters
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--finetune_wd', type=float, default=5e-2)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_batch_size', type=int, default=32) # Adjusted default

    # Selection strategy control
    parser.add_argument('--only_deterministic_strategies', action='store_true', help="Run only non-random strategies (entropy, bald)")
    parser.add_argument('--only_random_strategies', action='store_true', help="Run only random strategies")
    parser.add_argument('--without_epig', action='store_true', help="Skip EPIG strategy")
    parser.add_argument('--only_epig', action='store_true', help="Run only EPIG strategy")

    # EPIG parameters
    parser.add_argument('--epig_lr', type=float, default=1e-4, help="Learning rate for EPIG online updates")
    parser.add_argument('--epig_hessian_update_scale', type=float, default=10.0, help="Scaling factor beta for EPIG Hessian updates")
    parser.add_argument('--epig_num_samples', type=int, default=100, help="Number of MC samples for EPIG approximation")

    # k-NN parameters
    parser.add_argument('--k_nearest', type=int, default=1, help="Value 'k' for k-NN selection (neighbors per test point)")
    parser.add_argument('--knn_method', type=str, default='wasserstein', choices=['cosine', 'wasserstein'], help="Distance/Similarity metric for k-NN")
    parser.add_argument('--knn_buffersize', type=int, default=150, help="Buffer size for k-NN to ensure uniqueness")


    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    # Basic validation
    if args.only_deterministic_strategies and args.only_random_strategies:
        raise ValueError("Cannot set both --only_deterministic_strategies and --only_random_strategies")
    if args.only_epig and args.without_epig:
         raise ValueError("Cannot set both --only_epig and --without_epig")


    print("Starting Active Fine-tuning script with settings:")
    print(vars(args))

    main(
        model_str=args.model,
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,
        experiment_dir=args.experiment_dir,
        project_name=args.project_name,
        hessian_scale=args.hessian_scale,
        subset_size=args.subset_size,
        kmeans_clusters=args.kmeans_clusters, # Pass arg
        use_minibatch_kmeans=args.use_minibatch_kmeans,
        kmeans_batch_size=args.kmeans_batch_size,
        predictions_batch_size=args.predictions_batch_size,
        precompute_batch_size=args.precompute_batch_size,
        precompute_num_workers=args.precompute_num_workers,
        finetune_lr=args.finetune_lr,
        finetune_wd=args.finetune_wd,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,
        only_deterministic_strategies=args.only_deterministic_strategies,
        only_random_strategies=args.only_random_strategies,
        without_epig=args.without_epig,
        only_epig=args.only_epig,
        epig_lr=args.epig_lr,
        epig_hessian_update_scale=args.epig_hessian_update_scale,
        epig_num_samples=args.epig_num_samples,
        k_nearest=args.k_nearest,
        knn_method=args.knn_method,
        knn_buffersize=args.knn_buffersize, # Pass arg
        device=args.device,
    )

    print("Script finished.")