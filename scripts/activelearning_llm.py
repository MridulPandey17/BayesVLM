# --- START OF FILE activelearning_llm.py ---

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
import torch.nn.functional as F
import wandb
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_calibration_error,
)
from tqdm import tqdm
import google.generativeai as genai # <-- Add Gemini import
import os                            # <-- Add OS import
import time                          # <-- Add Time import
import re                            # <-- Add Regex import

# Assuming other bayesvlm modules are in the python path
from bayesvlm.vlm import CLIP, EncoderResult, ProbabilisticLogits
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

# --- Keep cluster_and_select_representatives, evaluate, finetune, run_knn ---
# (Copy these functions exactly from the previous activelearning.py EGL version)
def cluster_and_select_representatives(
    train_outputs: EncoderResult,
    k_clusters: int,
    device: str,
    seed: int = 0,
) -> tuple[torch.Tensor, EncoderResult]:
    # ... (function definition as provided before) ...
    print(f"    - Clustering training data into {k_clusters} clusters...")
    train_embeds_np = train_outputs.embeds.cpu().numpy()
    kmeans = KMeans(n_clusters=k_clusters, random_state=seed, n_init=10, verbose=0)
    kmeans.fit(train_embeds_np)
    centroids = kmeans.cluster_centers_
    print(f"    - Finding training samples closest to {k_clusters} centroids...")
    closest_indices, _ = pairwise_distances_argmin_min(centroids, train_embeds_np)
    representative_indices = torch.tensor(closest_indices, dtype=torch.long)
    representative_indices = torch.unique(representative_indices)
    print(f"    - Selected {len(representative_indices)} unique representative training samples.")
    representative_outputs = EncoderResult(
        embeds=train_outputs.embeds[representative_indices].clone(),
        activations=train_outputs.activations[representative_indices].clone(),
        residuals=train_outputs.residuals[representative_indices].clone(),
    )
    return representative_indices.to(device), representative_outputs

def evaluate(
    projection: torch.nn.Module,
    text_outputs: EncoderResult, # Should contain fixed text embeds
    clip: CLIP,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
):
    # ... (function definition as in EGL version) ...
    clip = clip.eval().to(device)
    projection = projection.eval().to(device)
    text_outputs = text_outputs.to(device) # Expects fixed embeds

    all_logits_mean = []
    all_logits_var = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for activations, residuals, lbls in loader:
            activations = activations.to(device)
            residuals = residuals.to(device)
            lbls = lbls.to(device)
            image_embeds = projection(activations) + residuals
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)

            # Use probabilistic forward pass
            prob_logits = clip(image_outputs, text_outputs) # text_outputs has fixed embeds

            all_logits_mean.append(prob_logits.mean.cpu())
            if prob_logits.var.ndim == 3:
                 all_logits_var.append(prob_logits.var.diagonal(dim1=-2, dim2=-1).cpu())
            else:
                 all_logits_var.append(prob_logits.var.cpu())
            all_labels.append(lbls.cpu())
            loss += prob_logits.cross_entropy(lbls, num_samples=0, reduction='sum').item()

    all_logits_mean = torch.cat(all_logits_mean, dim=0)
    all_logits_var = torch.cat(all_logits_var, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_logits_mean.argmax(dim=1) == all_labels).float().mean().item()
    acc_weighted = multiclass_accuracy(all_logits_mean, all_labels, num_classes=num_classes, average='weighted')
    ece = multiclass_calibration_error(all_logits_mean, all_labels, num_classes=num_classes)

    return dict(
        accuracy=acc,
        accuracy_weighted=acc_weighted.item(),
        ece=ece.item(),
        loss=loss / len(loader.dataset),
    )


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
    text_features: EncoderResult, # Full text features for eval setup
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
    epig_lr: float, # Still needed for wandb logging potentially
    epig_hessian_update_scale: float,
    epig_mc_samples: int = 100,
    knn_method: str = 'wasserstein',
):
    # ... (function definition as in EGL version, including wandb setup, fixed text embeds, training loop, evaluation loop) ...
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
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None: clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None: clip.logit_bias.data.requires_grad = False

    txt_projection = txt_projection.eval().to(device)
    for param in txt_projection.parameters(): param.requires_grad = False

    img_projection = img_projection.train().to(device)
    for param in img_projection.parameters(): param.requires_grad = True

    text_features = text_features.to(device)
    with torch.no_grad():
         text_embeds_fixed = txt_projection(text_features.activations)
         text_outputs_fixed = EncoderResult(embeds=text_embeds_fixed, activations=text_features.activations).to(device)


    train_ds = torch.utils.data.TensorDataset(image_features_train.activations.cpu(), image_features_train.residuals.cpu(), labels_train.cpu())
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_ds = torch.utils.data.TensorDataset(image_features_val.activations.cpu(), image_features_val.residuals.cpu(), labels_val.cpu())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)
    test_ds = torch.utils.data.TensorDataset(image_features_test.activations.cpu(), image_features_test.residuals.cpu(), labels_test.cpu())
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    params = list(img_projection.parameters())
    params = [p for p in params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    train_metrics = evaluate(img_projection, text_outputs_fixed, clip, train_loader, num_classes=num_classes, device=device)
    val_metrics = evaluate(img_projection, text_outputs_fixed, clip, val_loader, num_classes=num_classes, device=device)
    test_metrics = evaluate(img_projection, text_outputs_fixed, clip, test_loader, num_classes=num_classes, device=device)

    wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=0)
    wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=0)
    wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=0)

    best_val_loss = val_metrics.get('loss', float('inf'))
    best_test_metrics = test_metrics
    best_val_metrics = val_metrics
    best_projection = copy.deepcopy(img_projection)

    for epoch in range(epochs):
        img_projection.train()
        losses = []
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        for activations, residuals, lbls in pbar_train:
            optimizer.zero_grad()
            activations = activations.to(device)
            residuals = residuals.to(device)
            lbls = lbls.to(device)
            image_embeds = img_projection(activations) + residuals
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations)
            logits = clip.deterministic_forward(image_outputs, text_outputs_fixed)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar_train.set_postfix(loss=sum(losses) / len(losses))
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        img_projection.eval()
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
            best_projection = copy.deepcopy(img_projection)

        wandb.log({'train_loss_epoch': avg_loss}, step=epoch + 1)
        wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=epoch + 1)
        wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=epoch + 1)
        wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=epoch + 1)
        if best_test_metrics is not None:
            wandb.log({f'best_val_{k}': v for k, v in best_val_metrics.items()}, step=epoch + 1)
            wandb.log({f'best_test_{k}': v for k, v in best_test_metrics.items()}, step=epoch + 1)

    wandb.finish()
    return best_projection


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
        embeds_train = embeds_train.clone()
        embeds_test = embeds_test.clone()
        embeds_train.activations = torch.cat([embeds_train.activations, torch.ones_like(embeds_train.activations[:, :1])], dim=1)
        embeds_test.activations = torch.cat([embeds_test.activations, torch.ones_like(embeds_test.activations[:, :1])], dim=1)

    if method == 'cosine':
        return find_similar_samples_cosine(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    elif method == 'wasserstein':
        return find_similar_samples_wasserstein(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    else:
        raise ValueError(f"Unknown method {method}")

# --- LLM Helper Functions ---

def configure_gemini():
    """Configures the Gemini API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Try fetching from environment variable set by script caller if possible
        # For secure handling, environment variable is preferred
        try:
            import dotenv
            dotenv.load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
        except ImportError:
             print("dotenv not installed, cannot load .env file for API key.")
        except Exception as e:
             print(f"Error loading .env file: {e}")

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set or found in .env file.")
    genai.configure(api_key=api_key)
    print("Gemini API Key Configured.")


def generate_image_caption_placeholder(image_id, dataset_info) -> str:
    """
    Placeholder function to get/generate image description.
    Replace this with a real captioning method (e.g., BLIP, load pre-computed).
    """
    # print(f"Warning: Using placeholder caption for image {image_id}") # Reduce verbosity
    return f"Image content related to dataset '{dataset_info}' with ID {image_id}."


def call_gemini_api(prompt: str, model_name: str = "gemini-pro", retry_delay: int = 5, max_retries: int = 3) -> Optional[str]:
    """Makes an API call to Gemini with basic error handling and retries."""
    # Ensure API is configured before first call (or configure outside if preferred)
    if not genai.api_key:
        configure_gemini()

    model = genai.GenerativeModel(model_name)
    attempts = 0
    while attempts < max_retries:
        try:
            # Add safety settings if needed
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            response = model.generate_content(prompt, safety_settings=safety_settings)

            # Check for valid response text
            if response.parts:
                 return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            elif response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Warning: Prompt blocked by Gemini API. Reason: {response.prompt_feedback.block_reason}")
                 return f"BLOCKED: {response.prompt_feedback.block_reason}" # Return special string
            else:
                 # It might finish successfully but have no text content
                 print(f"Warning: Received empty or non-text response from Gemini for prompt: {prompt[:100]}...")
                 # print(f"Full Response obj: {response}") # Debugging
                 return None # Indicate failure or empty response

        except Exception as e:
            # Catch more specific API errors if possible using google.api_core.exceptions
            error_message = str(e)
            print(f"Error calling Gemini API: {error_message}. Attempt {attempts+1}/{max_retries}.")
            # Check for quota errors (often contain '429')
            if '429' in error_message or 'resource has been exhausted' in error_message.lower():
                 print(f"Quota possibly exceeded. Retrying in {retry_delay * (attempts + 1)}s...")
                 time.sleep(retry_delay * (attempts + 1)) # Exponential backoff might be better
            else:
                 time.sleep(retry_delay)
            attempts += 1
            if attempts == max_retries:
                print(f"Max retries reached for prompt: {prompt[:100]}...")
                return None # Indicate failure

def parse_llm_score(response_text: str, score_type: Literal['difficulty', 'value']) -> Optional[float]:
    """Placeholder function to parse the LLM score from its text response."""
    if response_text is None or response_text.startswith("BLOCKED"):
        print(f"Cannot parse score from None or blocked response: {response_text}")
        return None # Return None if blocked or failed

    try:
        # Try to find a number (integer or float) between 1 and 5
        # Handles patterns like "Score: 3", "Value: 4.5", "is a 2", etc.
        match = re.search(r'([1-5](?:\.\d+)?)', response_text)
        if match:
            score = float(match.group(1))
            # Clamp score just in case LLM returns something slightly outside range
            return max(1.0, min(5.0, score))
        else:
            print(f"Warning: Could not parse score (1-5) from response: {response_text[:100]}...")
            return None # Return None if parsing fails
    except Exception as e:
        print(f"Error parsing LLM score: {e} from response: {response_text[:100]}...")
        return None


# --- LLM Selection Function ---
@torch.no_grad() # Mostly involves API calls and parsing
def select_llm_score(
    image_outputs: EncoderResult, # Candidate features
    image_ids: torch.Tensor,      # Corresponding image identifiers
    image_class_ids: torch.Tensor,# Corresponding true labels (optional, maybe useful for prompt)
    text_outputs: EncoderResult,   # Class prompt features (not directly used, but context)
    class_prompts: list[str],      # Actual text of class prompts
    k: int,
    # device: str, # LLM operations mainly CPU/API
    llm_model_name: str,
    llm_score_type: Literal['difficulty', 'value'],
    dataset_name: str, # For caption placeholder
    batch_size: int = 5, # LLM calls are slow, use small batches
    rate_limit_delay: float = 1.1, # Seconds between batches (adjust based on RPM limit)
    default_score: float = 2.5, # Score to use if LLM fails/parsing fails
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selects candidates based on LLM scoring."""
    print(f"Selecting candidates using LLM ({llm_score_type} score)...")
    # Configure API key once at the start if needed
    if not genai.api_key:
        configure_gemini()

    num_candidates = image_outputs.activations.shape[0]
    # Store scores and corresponding original indices on CPU
    llm_scores_list = []
    original_indices_list = []


    candidate_indices_all = torch.arange(num_candidates)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(candidate_indices_all),
        batch_size=batch_size
    )

    pbar = tqdm(total=num_candidates, desc=f"LLM Scoring ({llm_score_type})", leave=False)
    processed_count = 0
    api_call_counter = 0
    start_time = time.time()

    for batch_idx_tensor, in loader:
        batch_indices = batch_idx_tensor.tolist()

        prompts_batch = []
        indices_in_batch = [] # Keep track of original index for score assignment

        # 1. Prepare prompts for the batch
        for _, original_idx in enumerate(batch_indices):
            img_id = image_ids[original_idx].item() # Get unique ID
            true_class_id = image_class_ids[original_idx].item()
            # Handle potential out-of-bounds class ID
            if true_class_id >= len(class_prompts):
                 print(f"Warning: Skipping sample with index {original_idx}, image ID {img_id}. Invalid class ID {true_class_id} >= num_classes {len(class_prompts)}.")
                 continue
            vlm_text_prompt = class_prompts[true_class_id]

            img_caption = generate_image_caption_placeholder(img_id, dataset_name)

            # --- Construct the actual prompt ---
            # Added instruction to ensure only the score is returned ideally
            format_instruction = " Respond with only the numerical score (1-5)."
            if llm_score_type == 'difficulty':
                 prompt = f"Image Description: '{img_caption}'. Text Prompt: '{vlm_text_prompt}'. How challenging (1=very easy, 5=very difficult) is it to definitively match this text prompt to this specific image, considering ambiguities, visual complexity, or subtle distinctions?{format_instruction}"
            elif llm_score_type == 'value': # Example for reasoning/value score
                 prompt = f"Image Description: '{img_caption}'. Text Prompt: '{vlm_text_prompt}'. This image-text pair was identified as potentially uncertain for a vision model. Plausible reasons include subtle visual cues, semantic ambiguity, unusual presentation, mismatch, or concept novelty. How valuable (1=low, 5=high) do you estimate this example would be for improving the model?{format_instruction}"
            else:
                 raise ValueError("Invalid llm_score_type")

            prompts_batch.append(prompt)
            indices_in_batch.append(original_idx) # Store the original index

        # 2. Call LLM API (sequentially for rate limiting)
        batch_scores = []
        for i, prompt in enumerate(prompts_batch):
            current_original_idx = indices_in_batch[i] # Get corresponding original index

            # Basic rate limiting check before call
            elapsed_time = time.time() - start_time
            expected_time_for_calls = api_call_counter * rate_limit_delay
            if elapsed_time < expected_time_for_calls:
                sleep_time = expected_time_for_calls - elapsed_time
                # print(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

            response_text = call_gemini_api(prompt, llm_model_name)
            api_call_counter += 1
            score = parse_llm_score(response_text, llm_score_type)

            if score is None:
                 print(f"Assigning default score {default_score} for index {current_original_idx}")
                 score = default_score # Use default if parsing/API failed

            batch_scores.append(score)

        # 3. Store scores and original indices for this batch
        llm_scores_list.extend(batch_scores)
        original_indices_list.extend(indices_in_batch)

        processed_count += len(batch_indices)
        pbar.update(len(batch_indices))

    pbar.close()
    print(f"LLM scoring finished. Total API calls: {api_call_counter}. Total time: {time.time() - start_time:.2f}s")


    # 4. Combine scores and select top k
    if not llm_scores_list:
         print("Warning: No LLM scores were successfully collected. Returning empty selection.")
         return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    llm_scores_tensor = torch.tensor(llm_scores_list, dtype=torch.float)
    original_indices_tensor = torch.tensor(original_indices_list, dtype=torch.long)

    # Sort by score (descending, higher score is better/more difficult/more valuable)
    sorted_scores, sorted_orig_indices_indices = torch.sort(llm_scores_tensor, descending=True)

    # Select top k original indices and their scores
    top_k_original_indices = original_indices_tensor[sorted_orig_indices_indices[:k]]
    top_k_scores = sorted_scores[:k]

    return top_k_original_indices.cpu(), top_k_scores.cpu()


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
    # Flags specific to this LLM script
    run_llm_difficulty: bool = False,
    run_llm_value: bool = False,


    # epig parameters (keep for comparison/logging)
    epig_lr: float = 1e-4,
    epig_hessian_update_scale: float = 10.0,
    epig_num_samples: int = 100,

    # knn parameters
    k_nearest: int = 1,
    knn_method: Literal['cosine', 'wasserstein'] = 'wasserstein',

    # LLM parameters
    llm_model_name: str = "gemini-pro",
    llm_batch_size: int = 5,
    llm_rate_limit_delay: float = 1.2, # >1s for 60 RPM

    device: str = 'cuda',
):
    run_dir = Path(experiment_dir) / dataset
    if not run_dir.exists():
        print(f"Creating run directory {run_dir}")
        run_dir.mkdir(parents=True)

    # --- Setup: Data, Model, Hessians, Predictions ---
    # (This section is identical to the activelearning.py EGL version)
    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    factory = DataModuleFactory(
        batch_size=precompute_batch_size, num_workers=precompute_num_workers,
        shuffle_train=False, train_transform=transform, test_transform=transform,
    )
    dm = factory.create(dataset)
    dm.setup()

    image_encoder, text_encoder, clip = load_model(model_str, device=device)
    image_encoder.freeze_all_layers()
    text_encoder.freeze_all_layers()
    if hasattr(clip, 'logit_scale') and clip.logit_scale is not None: clip.logit_scale.data.requires_grad = False
    if hasattr(clip, 'logit_bias') and clip.logit_bias is not None: clip.logit_bias.data.requires_grad = False

    print("[1] Precomputing features ...")
    # ... (Calls to precompute_image_features, precompute_text_features) ...
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
    # ... (Calls to load_hessians, optimize_prior_precision, compute_covariances, clip.set_covariances) ...
    A_img, B_img = load_hessians(la_dir=hessian_dir, tag='img', return_info=False)
    A_txt, B_txt, info = load_hessians(la_dir=hessian_dir, tag='txt', return_info=True)
    lambda_img = optimize_prior_precision(
        projection=image_encoder.vision_projection, A=A_img, B=B_img, lmbda_init=info['lambda_img'],
        n=hessian_scale, lr=1e-2, num_steps=500, device=device, retain_graph=False,
    ).item()
    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection, A=A_txt, B=B_txt, lmbda_init=info['lambda_txt'],
        n=hessian_scale, lr=1e-2, num_steps=500, device=device, retain_graph=False,
    ).item()
    covar_info = dict(lambda_img=lambda_img, lambda_txt=lambda_txt, n_img=hessian_scale, n_txt=hessian_scale)
    cov_img, cov_txt = compute_covariances(A_img.to(device), B_img.to(device), A_txt.to(device), B_txt.to(device), covar_info)
    clip.set_covariances(source_covariance=cov_img, target_covariance=cov_txt)


    print("[2] Making predictions ...")
    # ... (Calls to make_predictions for train, train_map, test, test_map) ...
    prob_logits_train = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False)
    prob_logits_train_map = make_predictions(clip=clip, image_outputs=image_outputs_train, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=True)
    # prob_logits_val needed? Only for fine-tuning eval.
    # prob_logits_val = make_predictions(clip=clip, image_outputs=image_outputs_val, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False)
    prob_logits_test = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=False)
    prob_logits_test_map = make_predictions(clip=clip, image_outputs=image_outputs_test, text_outputs=label_outputs, batch_size=predictions_batch_size, device=device, map_estimate=True)


    # --- Subset Directory Setup ---
    # Include LLM params in path name if running LLM strategies
    path_components = [
        f'subset_{subset_size}', f'k_{k_nearest}', f'n_{hessian_scale}', f'knn_{knn_method}',
    ]
    if run_llm_difficulty or run_llm_value:
        path_components.append(f'llm_{llm_model_name}')
    # Keep EPIG params if it might run alongside LLM
    if not without_epig and not only_deterministic_strategies and not only_random_strategies:
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
                 # Check KNN dict format (values are dicts with 'indices')
                 elif isinstance(data, list) and data and isinstance(data[0], dict) and 'indices' in data[0]:
                      # This might be the KNN format if loaded incorrectly - needs verification
                      pass # Assuming loaded format is handled by extract_test_train_indices
                 elif isinstance(data, list):
                      print(f"Warning: Unexpected list format for strategy {strategy}.")

    else:
        subset_indices_train = OrderedDict()

    # --- Determine which standard strategies to run (excluding LLM) ---
    # (Similar logic as before, but ensures LLM flags don't block standard ones unless intended)
    run_deterministic_non_epig = not only_random_strategies and not only_epig
    run_epig = not only_random_strategies and not without_epig
    run_random = not only_deterministic_strategies and not only_epig

    if only_deterministic_strategies: run_random = False; run_epig = False
    if only_random_strategies: run_deterministic_non_epig = False; run_epig = False
    if only_epig: run_deterministic_non_epig = False; run_random = False; run_epig = True

    # --- Standard Subset Selection Logic ---
    if run_deterministic_non_epig:
        # ... (Entropy MAP, Entropy Train, BALD Test - Copy from EGL version) ...
        entropy_key = 'entropy_map'
        print(f"    - {entropy_key} ...", flush=True)
        if entropy_key not in subset_indices_train:
            indices_test_sel, values_test_sel = select_topk(prob_logits_test_map, k=subset_size, variant='entropy', entropy_variant='map_alea', return_values=True)
            subset_indices_train[entropy_key] = run_knn(embeds_train=image_outputs_train, embeds_test=image_outputs_test, indices_test=indices_test_sel, values_test=values_test_sel, k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias)
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)

        entropy_train_key = 'entropy_map_train'
        print(f"    - {entropy_train_key} ...", flush=True)
        if entropy_train_key not in subset_indices_train:
            indices_train_sel, values_train_sel = select_topk(prob_logits_train_map, k=subset_size, variant='entropy', entropy_variant='map_alea', return_values=True)
            subset_indices_train[entropy_train_key] = {0: dict(score=0.0, indices=indices_train_sel.tolist(), similarities=values_train_sel.tolist())}
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)

        bald_key = 'bald_test'
        print(f"    - {bald_key} ...", flush=True)
        if bald_key not in subset_indices_train:
            indices_test_sel, values_test_sel = select_topk(prob_logits_test, k=subset_size, variant='exp_mutual_info', return_values=True, seed=0)
            subset_indices_train[bald_key] = run_knn(embeds_train=image_outputs_train, embeds_test=image_outputs_test, indices_test=indices_test_sel, values_test=values_test_sel, k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias)
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)

    if run_epig:
        # ... (EPIG KNN - Copy from EGL version) ...
        epig_key = 'epig_knn'
        print(f"    - {epig_key} ...", flush=True)
        if epig_key not in subset_indices_train:
            pooling_subsampling = 'knn_cosine' if knn_method == 'cosine' else 'knn_wasserstein'
            indices_epig, epig_scores = select_epig_online(label_features=label_outputs, pool_features=image_outputs_train, target_features=image_outputs_test, pool_class_ids=image_class_ids_train, image_projection=image_encoder.vision_projection, clip=clip, A_img=A_img, B_img=B_img, A_txt=A_txt, B_txt=B_txt, cov_info=covar_info, budget=subset_size, lr=epig_lr, hessian_update_scale=epig_hessian_update_scale, device=device, num_samples=epig_num_samples, seed=0, pool_max_size=40_000, target_max_size=20_000, pool_subsampling = pooling_subsampling, proj_has_bias=clip.source_projection_has_bias)
            subset_indices_train[epig_key] = {0: dict(score=0.0, indices=indices_epig, similarities=epig_scores)}
            with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)

    if run_random:
        # ... (Random on Test -> KNN, Random on Train Direct - Copy from EGL version) ...
        for i in range(5):
            random_test_key = f'random_on_test_{i}'
            print(f"    - {random_test_key} ...", flush=True)
            if random_test_key not in subset_indices_train:
                indices_random_test = select_random(prob_logits_test, k=subset_size, seed=i)
                subset_indices_train[random_test_key] = run_knn(embeds_train=image_outputs_train, embeds_test=image_outputs_test, indices_test=indices_random_test, values_test=torch.ones_like(indices_random_test.float()), k_nearest=k_nearest, source_covariance=clip.source_covariance, device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias)
                with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)

        for i in range(5):
            random_train_key = f'random_on_train_{i}'
            print(f"    - {random_train_key} ...", flush=True)
            if random_train_key not in subset_indices_train:
                indices_random_trivial_support = select_random(prob_logits_train, k=subset_size, seed=i)
                subset_indices_train[random_train_key] = {0: dict(score=0.0, indices=indices_random_trivial_support.tolist(), similarities=[1.0] * len(indices_random_trivial_support))}
                with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f)


    # --- LLM Subset Selection Logic ---
    # Add LLM Difficulty Score Selection
    if run_llm_difficulty:
         llm_diff_key = f'llm_difficulty_test'
         print(f"    - {llm_diff_key} ...", flush=True)
         if llm_diff_key not in subset_indices_train:
             if not hasattr(dm, 'class_prompts') or not dm.class_prompts:
                  raise ValueError("DatasetModule needs populated 'class_prompts' list for LLM selection.")

             indices_llm_diff_test, values_llm_diff_test = select_llm_score(
                 image_outputs=image_outputs_test, image_ids=image_ids_test, image_class_ids=image_class_ids_test,
                 text_outputs=label_outputs, class_prompts=dm.class_prompts,
                 k=subset_size,
                 llm_model_name=llm_model_name, llm_score_type='difficulty',
                 dataset_name=dataset, batch_size=llm_batch_size, rate_limit_delay=llm_rate_limit_delay,
             )

             subset_indices_train[llm_diff_key] = run_knn(
                 embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                 indices_test = indices_llm_diff_test.to(device), values_test = values_llm_diff_test.to(device),
                 k_nearest=k_nearest, source_covariance=clip.source_covariance,
                 device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
             )
             with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate

    # Add LLM Value Score Selection
    if run_llm_value:
         llm_val_key = f'llm_value_test'
         print(f"    - {llm_val_key} ...", flush=True)
         if llm_val_key not in subset_indices_train:
             if not hasattr(dm, 'class_prompts') or not dm.class_prompts:
                  raise ValueError("DatasetModule needs populated 'class_prompts' list for LLM selection.")
             print("Warning: Implementing LLM value scoring directly on test set, not as re-ranking of uncertain pool.")

             indices_llm_val_test, values_llm_val_test = select_llm_score(
                 image_outputs=image_outputs_test, image_ids=image_ids_test, image_class_ids=image_class_ids_test,
                 text_outputs=label_outputs, class_prompts=dm.class_prompts,
                 k=subset_size,
                 llm_model_name=llm_model_name, llm_score_type='value',
                 dataset_name=dataset, batch_size=llm_batch_size, rate_limit_delay=llm_rate_limit_delay,
             )

             subset_indices_train[llm_val_key] = run_knn(
                 embeds_train=image_outputs_train, embeds_test=image_outputs_test,
                 indices_test = indices_llm_val_test.to(device), values_test = values_llm_val_test.to(device),
                 k_nearest=k_nearest, source_covariance=clip.source_covariance,
                 device=device, method=knn_method, proj_has_bias=clip.source_projection_has_bias,
             )
             with open(subset_dir / 'subset_indices_train.json', 'w') as f: json.dump(subset_indices_train, f) # Save intermediate


    # --- Final save after all selections ---
    print("[3a] Saving final subset indices...")
    with open(subset_dir / 'subset_indices_train.json', 'w') as f:
        json.dump(subset_indices_train, f)


    # --- Fine-tuning Loop ---
    print("[4] Fine-tuning based on training subsets ...")
    # (Fine-tuning loop is identical to the EGL version - it just iterates through subset_indices_train)
    if not subset_indices_train:
         if json_path.exists():
             print(f"    - Loading existing subset indices from {json_path} for fine-tuning.")
             with open(json_path) as f: subset_indices_train = json.load(f)
         else:
             print("Error: No subset indices found or generated. Cannot fine-tune.")
             return

    for subset_key, indices_data in subset_indices_train.items():
        print(f"    - Fine-tuning on subset {subset_key} ...")
        try:
            if isinstance(indices_data, dict) and 0 in indices_data and 'indices' in indices_data[0]:
                 indices = indices_data[0]['indices'] # Direct selection
            elif isinstance(indices_data, dict):
                 indices = extract_test_train_indices(indices_data)['train'] # KNN selection
            else:
                 print(f"Warning: Unrecognized format for subset '{subset_key}'. Skipping."); continue

            indices = [int(i) for i in indices]
            if not indices: print(f"Warning: Empty index list for '{subset_key}'. Skipping."); continue

            max_train_idx = len(image_outputs_train) - 1
            valid_indices = [i for i in indices if 0 <= i <= max_train_idx]
            if len(valid_indices) != len(indices): print(f"Warning: Filtered {len(indices) - len(valid_indices)} OOB indices for '{subset_key}'."); indices = valid_indices
            if not indices: print(f"Warning: No valid indices left for '{subset_key}'. Skipping."); continue

            masked_image_features = image_outputs_train[indices]
            masked_class_ids = image_class_ids_train[indices]

            finetune_dir = subset_dir / subset_key
            checkpoint_path = finetune_dir / 'img_projection.pt'

            if checkpoint_path.exists(): print(f"    - Checkpoint exists at {checkpoint_path}. Skipping."); continue

            finetune_dir.mkdir(parents=True, exist_ok=True)
            img_projection_ft = copy.deepcopy(image_encoder.vision_projection)
            txt_projection_ft = copy.deepcopy(text_encoder.text_projection)

            def _selection_from_key(key: str):
                elements = key.split('_'); return '_'.join(elements[:-1]) if elements[-1].isdigit() and elements[0]=='random' else key

            best_img_projection = finetune(
                img_projection=img_projection_ft, txt_projection=txt_projection_ft, clip=clip,
                image_features_train=masked_image_features, labels_train=masked_class_ids,
                image_features_val=image_outputs_val, labels_val=image_class_ids_val,
                image_features_test=image_outputs_test, labels_test=image_class_ids_test,
                text_features=label_outputs, lr=finetune_lr, wd=finetune_wd, epochs=finetune_epochs,
                batch_size=finetune_batch_size, device=device, finetune_dir=finetune_dir,
                selection=_selection_from_key(subset_key), num_classes=len(dm.class_prompts),
                k_nearest=k_nearest, subset_size=len(indices), project_name=project_name, dataset=dataset,
                hessian_scale=hessian_scale, epig_lr=epig_lr, epig_hessian_update_scale=epig_hessian_update_scale,
                epig_mc_samples=epig_num_samples, knn_method=knn_method,
            )

            if best_img_projection is not None: torch.save(best_img_projection.state_dict(), checkpoint_path); print(f"    - Saved best projection to {checkpoint_path}")
            else: print(f"    - Fine-tuning did not produce a best projection for {subset_key}.")

        except Exception as e: print(f"ERROR during fine-tuning for subset {subset_key}: {e}"); import traceback; traceback.print_exc()


# --- Argparse and Main Call ---
if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    # --- Add ALL arguments from activelearning.py ---
    parser.add_argument('--model', type=str, default='clip-base')
    parser.add_argument('--dataset', type=str, default='homeoffice-da-clipart')
    parser.add_argument('--hessian_dir', type=str, default='hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K')
    parser.add_argument('--experiment_dir', type=str, default='experiments/active-finetuning-llm') # Changed default dir
    parser.add_argument('--project_name', type=str, default='active-finetuning-llm')   # Changed default project
    parser.add_argument('--subset_size', type=int, default=50)
    parser.add_argument('--hessian_scale', type=float, default=10)
    parser.add_argument('--predictions_batch_size', type=int, default=256)
    parser.add_argument('--precompute_batch_size', type=int, default=256)
    parser.add_argument('--precompute_num_workers', type=int, default=8)
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--finetune_wd', type=float, default=5e-2)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_batch_size', type=int, default=30)
    parser.add_argument('--only_deterministic_strategies', action='store_true', default=False)
    parser.add_argument('--only_random_strategies', action='store_true', default=False)
    parser.add_argument('--without_epig', action='store_true', default=False)
    parser.add_argument('--only_epig', action='store_true', default=False)
    parser.add_argument('--epig_lr', type=float, default=1e-4)
    parser.add_argument('--epig_hessian_update_scale', type=float, default=10.0)
    parser.add_argument('--k_nearest', type=int, default=1)
    parser.add_argument('--knn_method', type=str, default='wasserstein')
    parser.add_argument('--device', type=str, default='cuda')

    # --- Add LLM specific arguments ---
    parser.add_argument('--run_llm_difficulty', action='store_true', default=False, help='Run LLM difficulty scoring strategy')
    parser.add_argument('--run_llm_value', action='store_true', default=False, help='Run LLM value scoring strategy')
    parser.add_argument('--llm_model_name', type=str, default='gemini-1.5-flash-latest', help='Name of the Gemini model to use (e.g., gemini-pro, gemini-1.5-flash-latest)')
    parser.add_argument('--llm_batch_size', type=int, default=5, help='Batch size for LLM API calls')
    parser.add_argument('--llm_rate_limit_delay', type=float, default=1.1, help='Min delay (sec) between LLM API calls/batches (adjust based on model RPM)')


    args = parser.parse_args()

    # Pass all args to main
    main(**vars(args))

# --- END OF FILE activelearning_llm.py ---