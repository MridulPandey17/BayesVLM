# --- START OF FILE activelearning.py ---

import argparse
import copy
import json
from typing import Literal
from collections import OrderedDict
from pathlib import Path
from sklearn.cluster import KMeans # Or MiniBatchKMeans if dataset is large
from sklearn.metrics import pairwise_distances_argmin_min
import torch
import torch.utils.data
import torch.autograd as autograd        # <-- Added import
import torch.nn.functional as F          # <-- Added import (use F instead of full name)
import wandb
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_calibration_error,
)
from tqdm import tqdm

from bayesvlm.vlm import CLIP, EncoderResult, ProbabilisticLogits # <-- Added ProbabilisticLogits import

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

# --- Keep cluster_and_select_representatives function ---
def cluster_and_select_representatives(
    train_outputs: EncoderResult,
    k_clusters: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, EncoderResult]:
    # ... (function definition as provided before) ...
    print(f"    - Clustering training data into {k_clusters} clusters...")
    # KMeans works best on CPU with numpy
    train_embeds_np = train_outputs.embeds.cpu().numpy()

    # Use KMeans (or MiniBatchKMeans(n_clusters=k_clusters, random_state=seed, batch_size=1024, n_init=3) for large data)
    kmeans = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10, verbose=0)
    kmeans.fit(train_embeds_np)
    centroids = kmeans.cluster_centers_

    print(f"    - Finding training samples closest to {k_clusters} centroids...")
    # Find the index of the training sample closest to each centroid
    closest_indices, _ = pairwise_distances_argmin_min(centroids, train_embeds_np)
    representative_indices = torch.tensor(closest_indices, dtype=torch.long)

    # Ensure unique indices if multiple centroids map to the same closest point (rare)
    representative_indices = torch.unique(representative_indices)
    print(f"    - Selected {len(representative_indices)} unique representative training samples.")


    # Create a new EncoderResult containing only the representative features
    representative_outputs = EncoderResult(
        embeds=train_outputs.embeds[representative_indices].clone(),
        activations=train_outputs.activations[representative_indices].clone(),
        residuals=train_outputs.residuals[representative_indices].clone(),
    )

    return representative_indices.to(device), representative_outputs # Keep representatives on device


# --- Keep evaluate function ---
def evaluate(
    projection: torch.nn.Module,
    text_outputs: EncoderResult,
    clip: CLIP,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
):
    # ... (function definition as provided before) ...
    clip = clip.eval().to(device)
    projection = projection.eval().to(device)
    text_outputs = text_outputs.to(device)

    all_logits_mean = []
    all_logits_var = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for activations, residuals, lbls in loader:
            activations = activations.to(device)
            residuals = residuals.to(device)
            lbls = lbls.to(device)
            # Need to create EncoderResult within the loop for current projection
            image_embeds = projection(activations) + residuals
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)

            # Use the probabilistic forward pass (map_estimate=False is default)
            prob_logits = clip(image_outputs, text_outputs)

            all_logits_mean.append(prob_logits.mean.cpu())
            # Ensure variance is diagonal if needed by CE loss calc or metrics
            if prob_logits.var.ndim == 3:
                 all_logits_var.append(prob_logits.var.diagonal(dim1=-2, dim2=-1).cpu())
            else: # Already diagonal
                 all_logits_var.append(prob_logits.var.cpu())

            all_labels.append(lbls.cpu())

            # Calculate loss using probabilistic logits (probit approx)
            loss += prob_logits.cross_entropy(lbls, num_samples=0, reduction='sum').item() # Use num_samples=0 for probit


    all_logits_mean = torch.cat(all_logits_mean, dim=0)
    all_logits_var = torch.cat(all_logits_var, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_logits_mean.argmax(dim=1) == all_labels).float().mean().item()
    # Use MAP estimate for accuracy/ECE metrics as is standard
    acc_weighted = multiclass_accuracy(all_logits_mean, all_labels, num_classes=num_classes, average='weighted')
    ece = multiclass_calibration_error(all_logits_mean, all_labels, num_classes=num_classes)

    return dict(
        accuracy=acc,
        accuracy_weighted=acc_weighted.item(), # Ensure scalar
        ece=ece.item(), # Ensure scalar
        loss=loss / len(loader.dataset), # Average loss per sample
    )


# --- Keep finetune function ---
def finetune(
    img_projection: torch.nn.Module,
    txt_projection: torch.nn.Module,
    clip: CLIP,
    image_features_train: EncoderResult,
    labels_train: torch.Tensor,
    image_features_val: EncoderResult,
    labels_val: torch.Tensor,
    image_features_test: EncoderResult,
    labels_test: torch.Tensor,
    text_features: EncoderResult,
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
    epig_mc_samples: int = 100,
    knn_method: str = 'wasserstein',
):
    # ... (Wandb setup) ...
    wandb.init(project=project_name, dir=str(finetune_dir), reinit=True)
    wandb.config.update({
        'lr': lr, 'wd': wd, 'epochs': epochs, 'batch_size': batch_size,
        'selection': selection, 'subset_size': subset_size, 'k_nearest': k_nearest,
        'dataset': dataset, 'hessian_scale': hessian_scale, 'epig_lr': epig_lr,
        'epig_hessian_update_scale': epig_hessian_update_scale,
        'epig_mc_samples': epig_mc_samples, 'knn_method': knn_method,
    })
    wandb.run.name = finetune_dir.parent.name + '/' + finetune_dir.name


    clip = clip.eval().to(device)
    # Keep logit scale/bias fixed during fine-tuning
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None:
         clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None:
         clip.logit_bias.data.requires_grad = False

    # freeze projection layers for finetuning
    txt_projection = txt_projection.eval().to(device)
    for param in txt_projection.parameters():
        param.requires_grad = False

    # unfreeze projection layers for finetuning
    img_projection = img_projection.train().to(device)
    for param in img_projection.parameters():
        param.requires_grad = True

    # Precompute text embeds once (deterministic)
    text_features = text_features.to(device) # Keep full EncoderResult
    with torch.no_grad():
         text_embeds_fixed = txt_projection(text_features.activations) # Compute embeds
         text_outputs_fixed = EncoderResult(embeds=text_embeds_fixed, activations=text_features.activations) # Store


    train_ds = torch.utils.data.TensorDataset(image_features_train.activations.cpu(), image_features_train.residuals.cpu(), labels_train.cpu())
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    val_ds = torch.utils.data.TensorDataset(image_features_val.activations.cpu(), image_features_val.residuals.cpu(), labels_val.cpu())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    test_ds = torch.utils.data.TensorDataset(image_features_test.activations.cpu(), image_features_test.residuals.cpu(), labels_test.cpu())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    params = list(img_projection.parameters())
    params = [p for p in params if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # Initial evaluation
    train_metrics = evaluate(img_projection, text_outputs_fixed, clip, train_loader, num_classes=num_classes, device=device)
    val_metrics = evaluate(img_projection, text_outputs_fixed, clip, val_loader, num_classes=num_classes, device=device)
    test_metrics = evaluate(img_projection, text_outputs_fixed, clip, test_loader, num_classes=num_classes, device=device)

    wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=0)
    wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=0)
    wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=0)

    best_val_loss = val_metrics.get('loss', float('inf')) # Use initial val loss
    best_test_metrics = test_metrics
    best_val_metrics = val_metrics
    best_projection = copy.deepcopy(img_projection)

    for epoch in range(epochs):
        img_projection.train() # Ensure train mode
        losses = []
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for activations, residuals, lbls in pbar_train:
            optimizer.zero_grad()

            activations = activations.to(device)
            residuals = residuals.to(device)
            lbls = lbls.to(device)

            image_embeds = img_projection(activations) + residuals
            # Use fixed text embeds for standard CE loss calculation during training
            # The loss gradient should be based on the standard deterministic forward pass
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations) # No residuals needed here

            # Use deterministic forward for loss calculation
            logits = clip.deterministic_forward(image_outputs, text_outputs_fixed)

            loss = F.cross_entropy(logits, lbls)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            pbar_train.set_postfix(loss=sum(losses) / len(losses))

        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        # Evaluation after epoch
        img_projection.eval() # Set to eval mode for evaluation
        train_metrics = evaluate(img_projection, text_outputs_fixed, clip, train_loader, num_classes=num_classes, device=device)
        val_metrics = evaluate(img_projection, text_outputs_fixed, clip, val_loader, num_classes=num_classes, device=device)
        test_metrics = evaluate(img_projection, text_outputs_fixed, clip, test_loader, num_classes=num_classes, device=device)

        current_val_loss = val_metrics.get('loss', float('inf'))
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {current_val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")

        if current_val_loss <= best_val_loss:
            print(f"  -> New best validation loss found. Saving model.")
            best_val_loss = current_val_loss
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_projection = copy.deepcopy(img_projection) # Save the best model state

        # Log metrics for the current epoch
        wandb.log({'train_loss_epoch': avg_loss}, step=epoch + 1)
        wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=epoch + 1)
        wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=epoch + 1)
        wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=epoch + 1)
        # Log best metrics achieved so far
        if best_test_metrics is not None:
            wandb.log({f'best_val_{k}': v for k, v in best_val_metrics.items()}, step=epoch + 1)
            wandb.log({f'best_test_{k}': v for k, v in best_test_metrics.items()}, step=epoch + 1)

    wandb.finish() # Ensure wandb run is finished
    return best_projection

# --- Keep run_knn function ---
def run_knn(
    embeds_train: EncoderResult,
    embeds_test: EncoderResult,
    indices_test: torch.Tensor,
    values_test: torch.Tensor,
    k_nearest: int,
    device: str,
    source_covariance,
    method: str,
    proj_has_bias=False,
    ):
    # ... (function definition as provided before) ...
    if proj_has_bias:
        # Important: clone to avoid modifying the original data if called multiple times
        embeds_train = embeds_train.clone()
        embeds_test = embeds_test.clone()
        # Add bias term (ones) to activations
        embeds_train.activations = torch.cat([embeds_train.activations, torch.ones_like(embeds_train.activations[:, :1])], dim=1)
        embeds_test.activations = torch.cat([embeds_test.activations, torch.ones_like(embeds_test.activations[:, :1])], dim=1)


    if method == 'cosine':
        return find_similar_samples_cosine(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    elif method == 'wasserstein':
        return find_similar_samples_wasserstein(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    else:
        raise ValueError(f"Unknown method {method}")


# --- Define select_egl function here ---
@torch.no_grad() # Disable gradient calculation within this function where not explicitly needed
def select_egl(
    clip: CLIP,
    img_projection: torch.nn.Module,
    txt_projection: torch.nn.Module, # Needed for full forward pass
    image_outputs: EncoderResult,    # Candidates (e.g., test set features)
    text_outputs: EncoderResult,     # Class prompts features
    prob_logits: ProbabilisticLogits, # Precomputed posterior predictive for candidates
    k: int,
    device: str,
    batch_size: int = 32, # Batching for efficiency
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Selects candidates based on Expected Gradient Length (EGL).

    Computes EGL = E_{p(y|x,D)}[ ||∇_θ L((x,y); θ)|| ]
    where θ are the parameters of img_projection.

    Args:
        clip: The CLIP model instance (already configured with covariances).
        img_projection: The image projection layer (parameters to compute gradients for).
        txt_projection: The text projection layer.
        image_outputs: Features of the candidate pool (e.g., test set).
        text_outputs: Features of the class prompts.
        prob_logits: Precomputed posterior predictive distributions (mean/var/probs)
                     for the image_outputs candidates.
        k: Number of samples to select.
        device: Device to run computations on.
        batch_size: Batch size for processing candidates.

    Returns:
        Tuple of (top_k_indices, top_k_scores) on CPU.
    """
    print("Calculating EGL scores...")
    num_candidates = image_outputs.activations.shape[0]
    num_classes = text_outputs.activations.shape[0]
    egl_scores = torch.zeros(num_candidates, device='cpu') # Store scores on CPU

    # --- Ensure correct model states ---
    # Store initial states
    initial_img_proj_training_state = img_projection.training
    initial_txt_proj_training_state = txt_projection.training
    initial_clip_training_state = clip.training

    # Set models to appropriate modes
    img_projection = img_projection.train().to(device) # Need train mode for grads
    txt_projection = txt_projection.eval().to(device)  # Text projection fixed
    clip = clip.eval().to(device)               # CLIP forward pass fixed

    # Identify parameters for gradient calculation
    params_to_grad = [p for p in img_projection.parameters() if p.requires_grad]
    if not params_to_grad:
         print("Warning: No parameters in img_projection require gradients. Setting requires_grad=True temporarily for EGL.")
         for param in img_projection.parameters():
             param.requires_grad_(True)
         params_to_grad = list(img_projection.parameters())
         reset_grads_later = True
    else:
         reset_grads_later = False

    # --- Precompute fixed parts ---
    with torch.no_grad():
         # Compute text embeds once using the fixed text projection
         text_embeds_fixed = txt_projection(text_outputs.activations.to(device))
         text_outputs_fixed = EncoderResult(embeds=text_embeds_fixed, activations=text_outputs.activations).to(device)

    # --- Dataloader for candidates ---
    candidate_indices = torch.arange(num_candidates)
    # Use precomputed probabilities p(y|x,D) from prob_logits
    candidate_probs = prob_logits.probs.cpu() # Probs on CPU for dataloader

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            image_outputs.activations.cpu(),
            image_outputs.residuals.cpu(),
            candidate_probs,
            candidate_indices
        ),
        batch_size=batch_size
    )

    # --- Main EGL Calculation Loop ---
    pbar = tqdm(total=num_candidates, desc="EGL Calculation", leave=False)
    for batch_activations_cpu, batch_residuals_cpu, batch_probs_cpu, batch_indices_cpu in loader:
        batch_activations = batch_activations_cpu.to(device)
        batch_residuals = batch_residuals_cpu.to(device)
        batch_probs = batch_probs_cpu.to(device) # Shape: (batch_size, num_classes)
        batch_indices = batch_indices_cpu.tolist() # Use list for indexing later
        batch_size_current = batch_activations.shape[0]

        batch_egl = torch.zeros(batch_size_current, device=device)

        # Enable gradient tracking for this batch's activations
        batch_activations.requires_grad_(True)

        # Calculate image embeds for the batch using the *current* img_projection
        image_embeds = img_projection(batch_activations) + batch_residuals # Shape: (batch_size, embed_dim)

        # Iterate through each sample *in the batch*
        for i in range(batch_size_current):
            # Enable gradient calculation for this specific sample's forward pass
            with torch.enable_grad():
                img_embed_i = image_embeds[i].unsqueeze(0) # Shape: (1, embed_dim) - Keep grad history
                probs_i = batch_probs[i]                  # Shape: (num_classes,)

                # Compute deterministic logits for this image vs all text classes
                img_outputs_i = EncoderResult(embeds=img_embed_i, activations=None) # Activations not needed for deterministic forward
                all_logits_i = clip.deterministic_forward(img_outputs_i, text_outputs_fixed) # Shape: (1, num_classes)

                expected_norm = 0.0

                # Iterate through each possible class label y for the expectation
                for c in range(num_classes):
                    if probs_i[c] == 0: continue # Skip if probability is zero

                    # The loss for assuming the true class is 'c'
                    target_c = torch.tensor([c], device=device)
                    # Use the computed logits for this sample vs all classes
                    loss_ic = F.cross_entropy(all_logits_i, target_c)

                    # Compute gradient ONLY if loss is not zero (should always be positive)
                    if loss_ic > 0:
                        # Compute gradient of this loss w.r.t img_projection parameters
                        # Ensure grads are cleared before calculation for this specific (i, c) pair
                        img_projection.zero_grad(set_to_none=True)
                        # retain_graph=True is needed because we re-use parts of the graph (img_embed_i -> all_logits_i)
                        grads = autograd.grad(loss_ic, params_to_grad, retain_graph=True, allow_unused=False)

                        # Flatten and compute norm
                        flat_grads = []
                        for grad in grads:
                            if grad is not None:
                                flat_grads.append(grad.view(-1))
                        if flat_grads:
                             norm = torch.cat(flat_grads).norm(p=2)
                        else:
                             # This case should ideally not happen if params_to_grad is correct and loss > 0
                             print(f"Warning: No gradients computed for sample {i}, class {c}. Loss: {loss_ic.item()}")
                             norm = torch.tensor(0.0, device=device)
                    else:
                         norm = torch.tensor(0.0, device=device)

                    # Add to expectation, weighted by the posterior predictive probability
                    expected_norm += probs_i[c] * norm

            # Store EGL score for this candidate (sample i in the batch)
            batch_egl[i] = expected_norm

        # Store batch results back into the main tensor using original indices (on CPU)
        egl_scores[batch_indices] = batch_egl.cpu()
        pbar.update(batch_size_current)

    pbar.close()

    # --- Restore initial model states ---
    img_projection.train(initial_img_proj_training_state)
    txt_projection.train(initial_txt_proj_training_state)
    clip.train(initial_clip_training_state)

    # Reset requires_grad if we changed it
    if reset_grads_later:
        print("Resetting requires_grad=False for img_projection after EGL.")
        for param in img_projection.parameters():
            param.requires_grad_(False)

    # Clean up any remaining gradients
    img_projection.zero_grad(set_to_none=True)

    # --- Select top k ---
    # Ensure scores are on CPU before topk if not already
    egl_scores = egl_scores.cpu()
    top_k_scores, top_k_indices = torch.topk(egl_scores, k=k)

    return top_k_indices.cpu(), top_k_scores.cpu() # Return indices/scores on CPU


# --- Updated main function ---
def main(
    model_str: str,
    dataset: str,
    hessian_dir: str,

    experiment_dir: str,
    project_name: str,

    # experiment parameters
    hessian_scale: float,
    subset_size: int,

    # precompute parameters
    predictions_batch_size: int = 256,
    precompute_batch_size: int = 256,
    precompute_num_workers: int = 8,

    # fine-tuning parameters
    finetune_lr: float = 1e-5,
    finetune_wd: float = 5e-2,
    finetune_epochs: int = 100,
    finetune_batch_size: int = 30,

    # selection strategies to run
    only_deterministic_strategies: bool = False,
    only_random_strategies: bool = False,
    only_epig: bool = False,
    without_epig: bool = False,
    only_egl: bool = False, # <-- Added flag

    # epig parameters
    epig_lr: float = 1e-4,
    epig_hessian_update_scale: float = 10.0,
    epig_num_samples: int = 100,

    # knn parameters
    k_nearest: int = 1,
    knn_method: Literal['cosine', 'wasserstein'] = 'wasserstein',

    device: str = 'cuda',
):
    run_dir = Path(experiment_dir) / dataset
    if not run_dir.exists():
        print(f"Creating run directory {run_dir}")
        run_dir.mkdir(parents=True)

    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    factory = DataModuleFactory(
        batch_size=precompute_batch_size,
        num_workers=precompute_num_workers,
        shuffle_train=False,
        train_transform=transform,
        test_transform=transform,
    )
    dm = factory.create(dataset)
    dm.setup()

    # load / compute features
    image_encoder, text_encoder, clip = load_model(model_str, device=device)
    image_encoder.freeze_all_layers()
    text_encoder.freeze_all_layers()
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None:
         clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None:
         clip.logit_bias.data.requires_grad = False

    print("[1] Precomputing features ...")
    image_outputs_train, image_class_ids_train, image_ids_train = precompute_image_features(
        image_encoder=image_encoder, loader=dm.train_dataloader(),
        cache_dir=run_dir / 'base' / 'train', save_predictions=True,
    )
    image_outputs_val, image_class_ids_val, image_ids_val = precompute_image_features(
        image_encoder=image_encoder, loader=dm.val_dataloader(),
        cache_dir=run_dir / 'base' / 'val', save_predictions=True,
    )
    image_outputs_test, image_class_ids_test, image_ids_test = precompute_image_features(
        image_encoder=image_encoder, loader=dm.test_dataloader(),
        cache_dir=run_dir / 'base' / 'test', save_predictions=True,
    )
    label_outputs = precompute_text_features(
        text_encoder=text_encoder, class_prompts=dm.class_prompts,
        batch_size=precompute_batch_size, cache_dir=run_dir / 'base', save_predictions=True,
    )

    print("[1a] Loading Hessians and optimizing prior precision...")
    A_img, B_img = load_hessians(la_dir=hessian_dir, tag='img', return_info=False)
    A_txt, B_txt, info = load_hessians(la_dir=hessian_dir, tag='txt', return_info=True)

    lambda_img = optimize_prior_precision(
        projection=image_encoder.vision_projection, A=A_img, B=B_img,
        lmbda_init=info['lambda_img'], n=hessian_scale, lr=1e-2, num_steps=500, device=device, retain_graph=False, # Can set retain_graph=False here
    ).item()
    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection, A=A_txt, B=B_txt,
        lmbda_init=info['lambda_txt'], n=hessian_scale, lr=1e-2, num_steps=500, device=device, retain_graph=False,
    ).item()

    covar_info = dict(
        lambda_img=lambda_img, lambda_txt=lambda_txt,
        n_img=hessian_scale, n_txt=hessian_scale,
    )
    cov_img, cov_txt = compute_covariances(A_img.to(device), B_img.to(device), A_txt.to(device), B_txt.to(device), covar_info)
    clip.set_covariances(source_covariance=cov_img, target_covariance=cov_txt)

    print("[2] Making predictions ...")
    # Compute all necessary predictions once
    prob_logits_train = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False)
    prob_logits_train_map = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=True)
    prob_logits_val = make_predictions(clip=clip, image_outputs=image_outputs_val, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False) # Val needed? Only for fine-tuning eval.
    prob_logits_test = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False)
    prob_logits_test_map = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=True)


    # --- Subset Directory Setup ---
    path_components = [
        f'subset_{subset_size}',
        f'k_{k_nearest}',
        f'n_{hessian_scale}',
        f'knn_{knn_method}',
    ]
    # Only add EPIG params if EPIG might run
    if not without_epig and (only_epig or not only_deterministic_strategies and not only_random_strategies and not only_egl):
         path_components.extend([f'epig_lr_{epig_lr}', f'epig_update_{epig_hessian_update_scale}'])

    path = '_'.join(path_components)
    subset_dir = run_dir / path

    if not subset_dir.exists():
        subset_dir.mkdir(parents=True)

    print(f"[3] Creating training subsets in {subset_dir}...")
    json_path = subset_dir / 'subset_indices_train.json'
    if json_path.exists():
        print(f"    - Loading existing subset indices from {json_path}")
        with open(json_path) as f:
            subset_indices_train = json.load(f)
            # Convert inner list indices back to integers if loaded from JSON
            for strategy, data in subset_indices_train.items():
                 if isinstance(data, dict): # Handle structure like random_on_train/epig
                     if 'indices' in data.get(list(data.keys())[0], {}):
                         data[list(data.keys())[0]]['indices'] = [int(i) for i in data[list(data.keys())[0]]['indices']]
                 elif isinstance(data, list): # Handle old list format? Needs update if format changed.
                     print(f"Warning: Unexpected list format for strategy {strategy}, attempting conversion.")
                     # subset_indices_train[strategy] = [int(i) for i in data] # This is likely wrong for the new KNN dict format
    else:
        subset_indices_train = OrderedDict()


    # --- Determine which strategies to run based on flags ---
    run_deterministic_non_epig = not only_random_strategies and not only_epig and not only_egl
    run_epig = not only_random_strategies and not without_epig and not only_egl
    run_egl = not only_random_strategies and not only_epig
    run_random = not only_deterministic_strategies and not only_epig and not only_egl

    if only_deterministic_strategies:
        run_random = False
        run_epig = False # Assume EPIG isn't counted as deterministic here unless explicitly requested
    if only_random_strategies:
        run_deterministic_non_epig = False
        run_epig = False
        run_egl = False
    if only_epig:
        run_deterministic_non_epig = False
        run_random = False
        run_egl = False
        run_epig = True # Force EPIG true
    if only_egl:
        run_deterministic_non_epig = False
        run_random = False
        run_epig = False
        run_egl = True # Force EGL true


    # --- Subset Selection Logic ---

    # Check deterministic forward method for EGL
    if run_egl and not hasattr(clip, 'deterministic_forward'):
         raise NotImplementedError("CLIP class needs a 'deterministic_forward(img_enc_res, txt_enc_res)' method for EGL.")


    if run_deterministic_non_epig:
        # Aleatoric Entropy (MAP) on Test -> KNN Train
        entropy_key = 'entropy_map'
        print(f"    - {entropy_key} ...", flush=True)
        if entropy_key not in subset_indices_train:
            indices_test_sel, values_test_sel = select_topk(
                prob_logits_test_map, k=subset_size, variant='entropy',
                entropy_variant='map_alea', return_values=True,
            )
            subset_indices_train[entropy_key] = run_knn(
                embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                indices_test=indices_test_sel, values_test=values_test_sel,
                k_nearest=k_nearest, source_covariance=clip.source_covariance,
                device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
            )
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate

        # Aleatoric Entropy (MAP) on Train (Direct Selection)
        entropy_train_key = 'entropy_map_train'
        print(f"    - {entropy_train_key} ...", flush=True)
        if entropy_train_key not in subset_indices_train:
            indices_train_sel, values_train_sel = select_topk(
                prob_logits_train_map, k=subset_size, variant='entropy',
                entropy_variant='map_alea', return_values=True,
            )
            # Store directly selected train indices in the expected format
            subset_indices_train[entropy_train_key] = {
                0: dict(score=0.0, indices=indices_train_sel.tolist(), similarities=values_train_sel.tolist())
            }
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


        # BALD on Test -> KNN Train
        bald_key = 'bald_test'
        print(f"    - {bald_key} ...", flush=True)
        if bald_key not in subset_indices_train:
            indices_test_sel, values_test_sel = select_topk(
                prob_logits_test, k=subset_size, variant='exp_mutual_info', return_values=True, seed=0,
            )
            subset_indices_train[bald_key] = run_knn(
                embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                indices_test=indices_test_sel, values_test=values_test_sel,
                k_nearest=k_nearest, source_covariance=clip.source_covariance,
                device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
            )
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


    # EGL Selection
    if run_egl:
        egl_key = 'egl_test'
        print(f"    - {egl_key} ...", flush=True)
        if egl_key not in subset_indices_train:
            initial_img_projection = copy.deepcopy(image_encoder.vision_projection)
            initial_txt_projection = copy.deepcopy(text_encoder.text_projection)

            indices_egl_test, values_egl_test = select_egl(
                clip=clip,
                img_projection=initial_img_projection,
                txt_projection=initial_txt_projection,
                image_outputs=image_outputs_test,
                text_outputs=label_outputs,
                prob_logits=prob_logits_test, # Use full probabilistic logits
                k=subset_size,
                device=device,
                batch_size=predictions_batch_size, # Reuse prediction batch size or smaller
            )

            subset_indices_train[egl_key] = run_knn(
                embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                indices_test = indices_egl_test.to(device),
                values_test = values_egl_test.to(device),
                k_nearest=k_nearest, source_covariance=clip.source_covariance,
                device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
            )
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


    # EPIG Selection
    if run_epig:
        epig_key = 'epig_knn'
        print(f"    - {epig_key} ...", flush=True)
        if epig_key not in subset_indices_train:
            pooling_subsampling = 'knn_cosine' if knn_method == 'cosine' else 'knn_wasserstein'
            indices_epig, epig_scores = select_epig_online(
                label_features=label_outputs, pool_features=image_outputs_train,
                target_features=image_outputs_test, pool_class_ids=image_class_ids_train,
                image_projection=image_encoder.vision_projection, clip=clip,
                A_img=A_img, B_img=B_img, A_txt=A_txt, B_txt=B_txt, cov_info=covar_info,
                budget=subset_size, lr=epig_lr, hessian_update_scale=epig_hessian_update_scale,
                device=device, num_samples=epig_num_samples, seed=0,
                pool_max_size=40_000, target_max_size=20_000,
                pool_subsampling = pooling_subsampling, proj_has_bias=clip.source_projection_has_bias,
            )
            subset_indices_train[epig_key] = {
                0: dict(score=0.0, indices=indices_epig, similarities=epig_scores)
            }
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


    # Random Selection
    if run_random:
        # Random on Test -> KNN Train
        for i in range(5):
            random_test_key = f'random_on_test_{i}'
            print(f"    - {random_test_key} ...", flush=True)
            if random_test_key not in subset_indices_train:
                indices_random_test = select_random(prob_logits_test, k=subset_size, seed=i)
                subset_indices_train[random_test_key] = run_knn(
                    embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                    indices_test=indices_random_test, values_test=torch.ones_like(indices_random_test.float()), # Use dummy values
                    k_nearest=k_nearest, source_covariance=clip.source_covariance,
                    device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
                )
                with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate

        # Random on Train (Direct Selection)
        for i in range(5):
            random_train_key = f'random_on_train_{i}'
            print(f"    - {random_train_key} ...", flush=True)
            if random_train_key not in subset_indices_train:
                # Select train indices directly
                indices_random_trivial_support = select_random(prob_logits_train, k=subset_size, seed=i)
                subset_indices_train[random_train_key] = {
                    0: dict(score=0.0, indices=indices_random_trivial_support.tolist(), similarities=[1.0] * len(indices_random_trivial_support))
                }
                with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


    # --- Final save after all selections ---
    print("[3a] Saving final subset indices...")
    with open(subset_dir / 'subset_indices_train.json', 'w') as f:
        json.dump(subset_indices_train, f)

    print("[4] Fine-tuning based on training subsets ...")
    # Ensure subset_indices_train is loaded if it wasn't generated in this run
    if not subset_indices_train:
         if json_path.exists():
             print(f"    - Loading existing subset indices from {json_path} for fine-tuning.")
             with open(json_path) as f:
                 subset_indices_train = json.load(f)
         else:
             print("Error: No subset indices found or generated. Cannot proceed with fine-tuning.")
             return # Exit if no subsets defined


    for subset_key, indices_data in subset_indices_train.items():
        print(f"    - Fine-tuning on subset {subset_key} ...")
        try:
            # Handle both direct index lists and KNN dict format
            if isinstance(indices_data, dict) and 0 in indices_data and 'indices' in indices_data[0]:
                 # Direct selection (e.g., random_on_train, epig_knn, entropy_map_train)
                 indices = indices_data[0]['indices']
            elif isinstance(indices_data, dict):
                 # KNN selection (e.g., bald_test, random_on_test, entropy_map, egl_test)
                 indices = extract_test_train_indices(indices_data)['train']
            else:
                 print(f"Warning: Unrecognized format for indices_data for key '{subset_key}'. Skipping.")
                 continue

            # Ensure indices are integers
            indices = [int(i) for i in indices]

            if not indices:
                print(f"Warning: Empty index list for subset '{subset_key}'. Skipping fine-tuning.")
                continue

            # Ensure indices are within bounds
            max_train_idx = len(image_outputs_train) - 1
            valid_indices = [i for i in indices if 0 <= i <= max_train_idx]
            if len(valid_indices) != len(indices):
                 print(f"Warning: Filtered out {len(indices) - len(valid_indices)} out-of-bounds indices for subset '{subset_key}'.")
                 indices = valid_indices
                 if not indices:
                      print(f"Warning: No valid indices left for subset '{subset_key}' after filtering. Skipping.")
                      continue


            masked_image_features = image_outputs_train[indices]
            masked_class_ids = image_class_ids_train[indices]
            # masked_image_ids = image_ids_train[indices] # Not used in finetune function

            finetune_dir = subset_dir / subset_key
            checkpoint_path = finetune_dir / 'img_projection.pt'

            # Check if checkpoint already exists
            if checkpoint_path.exists():
                 print(f"    - Checkpoint already exists at {checkpoint_path}. Skipping fine-tuning.")
                 continue # Skip to next subset if already done

            finetune_dir.mkdir(parents=True, exist_ok=True)

            # Create fresh copies of projections for each fine-tuning run
            img_projection_ft = copy.deepcopy(image_encoder.vision_projection)
            txt_projection_ft = copy.deepcopy(text_encoder.text_projection)

            def _selection_from_key(key: str):
                elements = key.split('_')
                # if last element is a number, it is a seed
                if elements[-1].isdigit() and elements[0]=='random': # Be more specific for random seeds
                    return '_'.join(elements[:-1])
                return key

            # Run fine-tuning
            best_img_projection = finetune(
                img_projection=img_projection_ft,
                txt_projection=txt_projection_ft,
                clip=clip, # Pass the main clip model (fixed text projection, fixed logit scale)
                image_features_train=masked_image_features,
                labels_train=masked_class_ids,
                image_features_val=image_outputs_val,
                labels_val=image_class_ids_val,
                image_features_test=image_outputs_test,
                labels_test=image_class_ids_test,
                text_features=label_outputs, # Pass full text features for eval
                lr=finetune_lr, wd=finetune_wd, epochs=finetune_epochs,
                batch_size=finetune_batch_size, device=device,
                finetune_dir=finetune_dir, selection=_selection_from_key(subset_key),
                num_classes=len(dm.class_prompts), k_nearest=k_nearest,
                subset_size=len(indices), # Log actual subset size used
                project_name=project_name, dataset=dataset, hessian_scale=hessian_scale,
                epig_lr=epig_lr, epig_hessian_update_scale=epig_hessian_update_scale,
                epig_mc_samples=epig_num_samples, knn_method=knn_method,
            )

            # Save the best projection weights from fine-tuning
            if best_img_projection is not None:
                 torch.save(best_img_projection.state_dict(), checkpoint_path)
                 print(f"    - Saved best projection to {checkpoint_path}")
            else:
                 print(f"    - Fine-tuning did not produce a best projection for {subset_key}.")

        except Exception as e:
            print(f"ERROR during fine-tuning for subset {subset_key}: {e}")
            import traceback
            traceback.print_exc()
            # Optionally: continue to the next subset


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='clip-base')
    parser.add_argument('--dataset', type=str, default='homeoffice-da-clipart')
    parser.add_argument('--hessian_dir', type=str, default='hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K')

    parser.add_argument('--experiment_dir', type=str, default='experiments/active-finetuning')
    parser.add_argument('--project_name', type=str, default='active-finetuning')

    # experiment parameters
    parser.add_argument('--subset_size', type=int, default=50)
    parser.add_argument('--hessian_scale', type=float, default=10)

    # precompute parameters
    parser.add_argument('--predictions_batch_size', type=int, default=256)
    parser.add_argument('--precompute_batch_size', type=int, default=256)
    parser.add_argument('--precompute_num_workers', type=int, default=8)

    # fine-tuning parameters
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--finetune_wd', type=float, default=5e-2)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_batch_size', type=int, default=30)

    # which selection strategies to run
    parser.add_argument('--only_deterministic_strategies', action='store_true', default=False)
    parser.add_argument('--only_random_strategies', action='store_true', default=False)
    parser.add_argument('--without_epig', action='store_true', default=False)
    parser.add_argument('--only_epig', action='store_true', default=False)
    parser.add_argument('--only_egl', action='store_true', default=False) # <-- Added flag

    # epig parameters
    parser.add_argument('--epig_lr', type=float, default=1e-4)
    parser.add_argument('--epig_hessian_update_scale', type=float, default=10.0)

    # knn parameters
    parser.add_argument('--k_nearest', type=int, default=1)
    parser.add_argument('--knn_method', type=str, default='wasserstein')

    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    main(
        model_str=args.model,
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,

        experiment_dir=args.experiment_dir,
        project_name=args.project_name,

        # experiment parameters
        hessian_scale=args.hessian_scale,
        subset_size=args.subset_size,

        # precompute parameters
        predictions_batch_size=args.predictions_batch_size,
        precompute_batch_size=args.precompute_batch_size,
        precompute_num_workers=args.precompute_num_workers,

        # finetuning parameters
        finetune_lr=args.finetune_lr,
        finetune_wd=args.finetune_wd,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,

        # selection strategies to run
        only_deterministic_strategies=args.only_deterministic_strategies,
        only_random_strategies=args.only_random_strategies,
        without_epig=args.without_epig,
        only_epig=args.only_epig,
        only_egl=args.only_egl, # <-- Pass flag

        # epig parameters
        epig_lr=args.epig_lr,
        epig_hessian_update_scale=args.epig_hessian_update_scale,

        # knn parameters
        k_nearest=args.k_nearest,
        knn_method=args.knn_method,

        device=args.device,
    )
# --- END OF FILE activelearning.py ---