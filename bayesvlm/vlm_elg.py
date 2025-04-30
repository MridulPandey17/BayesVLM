# --- START OF FILE vlm.py ---

from typing import Optional
from dataclasses import dataclass

import torch
import math
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPModel,
    SiglipTextModel,
    SiglipVisionModel,
    SiglipModel,
)

from bayesvlm.hessians import KroneckerFactorizedCovariance


PROJECTION_DIM = {
    'laion/CLIP-ViT-B-32-laion2B-s34B-b79K': 512,
    'laion/CLIP-ViT-L-14-laion2B-s32B-b82K': 768,
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1024,
    # Add Siglip projection dims if needed, e.g.:
    'google/siglip-base-patch16-256': 768,
    'google/siglip-large-patch16-256': 1024,
}

@dataclass
class EncoderResult:
    embeds: torch.Tensor
    activations: torch.Tensor
    residuals: torch.Tensor

    def __init__(self, embeds, activations, residuals=None):
        self.embeds = embeds
        self.activations = activations
        self.residuals = residuals if residuals is not None else torch.zeros_like(embeds)

    def clone(self):
        return EncoderResult(
            embeds=self.embeds.clone(),
            activations=self.activations.clone(),
            residuals=self.residuals.clone(),
        )

    def to(self, device):
        self.embeds = self.embeds.to(device)
        self.activations = self.activations.to(device)
        self.residuals = self.residuals.to(device)
        return self

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        if isinstance(idx, (list, torch.Tensor)):
            return EncoderResult(
                embeds=self.embeds[idx],
                activations=self.activations[idx],
                residuals=self.residuals[idx],
            )
        # Allow direct indexing for tuple unpacking
        return self.embeds[idx], self.activations[idx], self.residuals[idx]

@dataclass
class ProbabilisticLogits:
    mean: torch.Tensor
    var: torch.Tensor # Variance can be diagonal [N, Cl] or full covariance [N, Cl, Cl]

    @property
    def probs(self) -> torch.Tensor:
        """Approximated probabilities using probit approximation."""
        if self.var.ndim == 2: # Diagonal variance
            variance = self.var
        elif self.var.ndim == 3: # Full covariance
            variance = self.var.diagonal(dim1=-2, dim2=-1)
        else:
            raise ValueError("Invalid variance tensor shape.")

        scaled_mean = self.mean / torch.sqrt(1 + torch.pi / 8 * variance)
        return torch.nn.functional.softmax(scaled_mean, dim=-1)

    def softmax(self, dim=-1, num_samples=400, chunk_size=10000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        if num_samples == 0:
            # Return probit approximation directly
            return self.probs

        # Monte Carlo Sampling
        probas = torch.zeros_like(self.mean)

        if self.var.ndim == 2: # Diagonal variance
             std = torch.sqrt(self.var)
             for _ in range(num_samples):
                 eps = torch.randn_like(std) * std
                 probas += torch.nn.functional.softmax(self.mean + eps, dim=dim)

        elif self.var.ndim == 3: # Full covariance
            num_chunks = math.ceil(self.mean.shape[0] / chunk_size)
            mean_chunks = torch.chunk(self.mean, num_chunks, dim=0)
            var_chunks = torch.chunk(self.var, num_chunks, dim=0)

            probas_list = []
            pbar_desc = "Softmax Sampling"
            pbar = tqdm(zip(mean_chunks, var_chunks), total=num_chunks, desc=pbar_desc, leave=False)
            for mean_chunk, var_chunk in pbar:
                try:
                     dist = torch.distributions.MultivariateNormal(mean_chunk, covariance_matrix=var_chunk)
                except ValueError as e:
                     print(f"Error creating MVN distribution: {e}. Adding small jitter to covariance.")
                     # Add jitter to diagonal for numerical stability
                     jitter = torch.eye(var_chunk.shape[-1], device=var_chunk.device) * 1e-6
                     dist = torch.distributions.MultivariateNormal(mean_chunk, covariance_matrix=var_chunk + jitter)

                probas_chunk = torch.zeros_like(mean_chunk)
                for _ in range(num_samples):
                    sample = dist.sample()
                    probas_chunk += torch.nn.functional.softmax(sample, dim=dim)
                probas_list.append(probas_chunk)
            probas = torch.cat(probas_list, dim=0)

        return probas / num_samples

    def sample_probas(self, num_samples: int, seed=None):
        """
        Sample from the distribution and return the softmax probabilities.

        Args:
            num_samples (int): Number of samples to draw from the distribution.

        Returns:
            torch.Tensor: [N, num_samples, num_classes]
        """

        if seed is not None:
            torch.manual_seed(seed)

        if self.var.ndim == 2: # Diagonal variance
            std = torch.sqrt(self.var)
            # Shape: (num_samples, N, Cl) -> permute to (N, num_samples, Cl)
            samples = torch.randn((num_samples,) + self.mean.shape, device=self.mean.device) * std + self.mean
            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)

        elif self.var.ndim == 3: # Full covariance
            try:
                 dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.var)
            except ValueError as e:
                 print(f"Error creating MVN distribution: {e}. Adding small jitter to covariance.")
                 jitter = torch.eye(self.var.shape[-1], device=self.var.device) * 1e-6
                 dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.var + jitter)

            # Shape: (num_samples, N, Cl) -> permute to (N, num_samples, Cl)
            samples = dist.sample((num_samples, ))
            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)

        else:
            raise ValueError("Invalid variance tensor shape.")


    def expected_aleatoric_entropy(self, num_samples=400, dim=-1, seed=None):
        entropy = 0

        # Re-use sampling logic from sample_probas for consistency
        # Shape: (N, num_samples, Cl)
        sampled_probas = self.sample_probas(num_samples, seed=seed)

        # Calculate entropy for each sample's probability distribution
        # Shape: (N, num_samples)
        entropies_per_sample = -(sampled_probas * torch.log(sampled_probas + 1e-9)).sum(dim=dim) # Add epsilon for stability

        # Average entropy across samples
        # Shape: (N,)
        entropy = entropies_per_sample.mean(dim=1)

        return entropy

    def __getitem__(self, idx):
        return ProbabilisticLogits(
            mean=self.mean[idx],
            var=self.var[idx],
        )

    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self

    def detach(self):
        return ProbabilisticLogits(
            mean=self.mean.detach(),
            var=self.var.detach(),
        )

    def cross_entropy(self, target, num_samples=400, reduction='sum', seed=None):
        if num_samples == 0:
            # Use probit approximation for CE loss
            probit_probs = self.probs
            # Compute loss using log_softmax for numerical stability
            log_probs = torch.log_softmax(probit_probs, dim=-1) # Use probit probs here
            # NLLLoss expects log-probabilities
            return torch.nn.functional.nll_loss(log_probs, target, reduction=reduction)


        loss = 0
        # Re-use sampling logic
        # Shape: (N, num_samples, Cl)
        sampled_probas = self.sample_probas(num_samples, seed=seed)

        # Average the loss over samples
        # Need to compute loss for each sample and then average
        total_loss = 0.0
        num_items = target.numel() if reduction == 'sum' else 1 # Adjust divisor based on reduction

        # Iterate through samples
        for s in range(num_samples):
             # Get probabilities for this sample: shape (N, Cl)
             probs_s = sampled_probas[:, s, :]
             # Calculate log probabilities for NLLLoss
             log_probs_s = torch.log(probs_s + 1e-9) # Add epsilon
             # Calculate loss for this sample
             loss_s = torch.nn.functional.nll_loss(log_probs_s, target, reduction=reduction)
             total_loss += loss_s

        return total_loss / (num_samples * num_items) if reduction == 'mean' else total_loss / num_samples


    def clone(self):
        return ProbabilisticLogits(
            mean=self.mean.clone(),
            var=self.var.clone(),
        )

class CLIPTextEncoder(torch.nn.Module):
    def __init__(
            self,
            text_model: CLIPTextModelWithProjection,
            tokenizer: AutoTokenizer,
        ):
        super().__init__()
        self.text_encoder = text_model.text_model
        self.text_projection = text_model.text_projection
        self.tokenizer = tokenizer
        self.device = text_model.device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if projection_dim is None:
            projection_dim = PROJECTION_DIM[model_name]

        text_model = CLIPTextModelWithProjection.from_pretrained(model_name, projection_dim=projection_dim)
        model = cls(text_model, tokenizer)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        state_dict = torch.load(path)
        self.text_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True

        for layer in self.text_encoder.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, batch, return_activations=False):
        texts = batch['text']
        text_input = self.tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_outputs = self.text_encoder(**text_input)
        text_pooled_output = text_outputs[1]
        text_embeds = self.text_projection(text_pooled_output)

        if return_activations:
            return EncoderResult(embeds=text_embeds, activations=text_pooled_output)

        return text_embeds

class CLIPImageEncoder(torch.nn.Module):
    def __init__(
            self,
            vision_model: CLIPVisionModelWithProjection,
        ):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.vision_projection = vision_model.visual_projection
        self.device = vision_model.device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if projection_dim is None:
            projection_dim = PROJECTION_DIM[model_name]

        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            projection_dim=projection_dim,
        )
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        model.device = device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        state_dict = torch.load(path)
        self.vision_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True

        for layer in self.vision_encoder.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True


    def forward(self, batch, return_activations=False):
        images = batch['image']
        image_input = dict(pixel_values=images.to(self.device))
        image_outputs = self.vision_encoder(**image_input)
        image_pooled_output = image_outputs[1]
        image_embeds = self.vision_projection(image_pooled_output)

        if return_activations:
            return EncoderResult(embeds=image_embeds, activations=image_pooled_output)

        return image_embeds

# ... (Keep SiglipTextEncoder and SiglipImageEncoder classes as they are) ...
class SiglipTextEncoder(torch.nn.Module):
    def __init__(
        self,
        model: SiglipTextModel,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self._siglip_text_transformer = model.text_model
        self.text_projection = model.text_model.head
        self.tokenizer = tokenizer
        self.device = model.device # Store device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = SiglipTextModel.from_pretrained(model_name)
        model = cls(model, tokenizer)
        model = model.to(device) if device is not None else model
        # Ensure device attribute is set correctly
        if device is not None:
             model.device = torch.device(device)
        else:
             # Infer device from model parameters if not specified
             model.device = next(model.parameters()).device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        state_dict = torch.load(path)
        self.text_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True

        for layer in self._siglip_text_transformer.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, batch, return_activations=False):
        texts = batch['text']
        # Ensure tokenizer sends inputs to the correct device
        text_input = self.tokenizer(text=texts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        hidden_states = self._siglip_text_transformer.embeddings(input_ids=text_input.input_ids, attention_mask=text_input.attention_mask) # Pass explicit args if needed
        encoder_outputs = self._siglip_text_transformer.encoder(hidden_states, attention_mask=text_input.attention_mask) # Pass attention mask
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self._siglip_text_transformer.final_layer_norm(last_hidden_state)
        # Correct pooling for Siglip (usually CLS token at index 0)
        pooled_output = last_hidden_state[:, 0, :]

        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            # Return the pooled output as activations
            return EncoderResult(embeds=text_embeds, activations=pooled_output)

        return text_embeds

class SiglipVisionEncoderWithoutProjection(torch.nn.Module):
    def __init__(
        self,
        model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.vision_model.embeddings(pixel_values)
        encoder_outputs = self.vision_model.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)

        batch_size = last_hidden_state.shape[0]
        # Ensure probe is on the correct device
        probe = self.vision_model.head.probe.repeat(batch_size, 1, 1).to(last_hidden_state.device)
        # Ensure keys/values are passed correctly if needed by attention implementation
        last_hidden_state = self.vision_model.head.attention(query=probe, key=last_hidden_state, value=last_hidden_state)[0]


        residual = last_hidden_state
        last_hidden_state = self.vision_model.head.layernorm(last_hidden_state)
        mlp = self.vision_model.head.mlp

        last_hidden_state = mlp.fc1(last_hidden_state)
        last_hidden_state = mlp.activation_fn(last_hidden_state)

        return last_hidden_state, residual

class SiglipImageEncoder(torch.nn.Module):
    def __init__(
        self,
        vision_model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_encoder = SiglipVisionEncoderWithoutProjection(vision_model)
        self.vision_projection = vision_model.vision_model.head.mlp.fc2
        self.device = vision_model.device # Store device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        vision_model = SiglipVisionModel.from_pretrained(model_name)
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        # Ensure device attribute is set correctly
        if device is not None:
             model.device = torch.device(device)
        else:
             model.device = next(model.parameters()).device
        return model

    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return

        if path is None:
            raise ValueError("Either path or state_dict must be provided.")

        state_dict = torch.load(path)
        self.vision_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True

        # Adapt layer access for Siglip vision transformer structure if different from CLIP
        # This assumes a similar .encoder.layers structure; adjust if needed.
        if hasattr(self.vision_encoder.vision_model, 'encoder') and hasattr(self.vision_encoder.vision_model.encoder, 'layers'):
             for layer in self.vision_encoder.vision_model.encoder.layers[-k_last_layers:]:
                 layer.train()
                 for param in layer.parameters():
                     param.requires_grad = True
        else:
            print("Warning: Could not find standard encoder layers for gradient enabling in SiglipImageEncoder.")


    def forward(self, batch, return_activations=False):
        images = batch['image']
        image_input = images.to(self.device)
        activations, residuals = self.vision_encoder(image_input)

        # Siglip pooling often takes the first token (probe query) result
        activations = activations[:, 0]
        residuals = residuals[:, 0]

        image_embeds = self.vision_projection(activations) + residuals

        if return_activations:
            return EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)

        return image_embeds


class CLIP(torch.nn.Module):
    source_projection_has_bias = False
    target_projection_has_bias = False

    def __init__(
        self,
        logit_scale: float,
        logit_bias: float = 0,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([], device=device) * logit_scale)
        # CLIP typically doesn't have a logit bias, initialize to 0 if provided but maybe remove later
        self.logit_bias = torch.nn.Parameter(torch.ones([], device=device) * logit_bias) if logit_bias is not None else None
        self.source_covariance = source_covariance
        self.target_covariance = target_covariance

    @property
    def device(self):
        return self.logit_scale.data.device

    def set_covariances(
        self,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
    ):
        self.source_covariance = KroneckerFactorizedCovariance(
            A_inv=source_covariance.A_inv.clone().to(self.device),
            B_inv=source_covariance.B_inv.clone().to(self.device),
        ) if source_covariance is not None else None

        self.target_covariance = KroneckerFactorizedCovariance(
            A_inv=target_covariance.A_inv.clone().to(self.device),
            B_inv=target_covariance.B_inv.clone().to(self.device),
        ) if target_covariance is not None else None

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        clip = CLIPModel.from_pretrained(model_name)
        model = cls(
            logit_scale=clip.logit_scale.item(),
            logit_bias=None, # Standard CLIP doesn't use this
        )
        model = model.to(device) if device is not None else model
        return model

    def deterministic_forward(
            self,
            source_results: EncoderResult,
            target_results: EncoderResult,
        ):
        """ Computes standard (non-probabilistic) logits from embeddings. """
        source_embeds = source_results.embeds
        target_embeds = target_results.embeds
        # normalize
        source_embeds = torch.nn.functional.normalize(source_embeds, p=2, dim=-1)
        target_embeds = torch.nn.functional.normalize(target_embeds, p=2, dim=-1)

        # cosine similarity
        logits = torch.matmul(source_embeds, target_embeds.t()) * self.logit_scale.exp()
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits

    def _compute_logits( # Alias for deterministic forward for internal consistency if needed
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ):
        return self.deterministic_forward(EncoderResult(embeds=source_embeds, activations=None),
                                         EncoderResult(embeds=target_embeds, activations=None))


    def _compute_probabilistic_logits_smith(
        self,
        source_results: EncoderResult,
        target_results: EncoderResult,
        compute_covariance: bool = False,
    ):
        """
        This function compute the expected value and variance of the cosine similarity between two probabilistic embeddings.
        The derivation adopts the approach by Smith et al. (2023).
        """

        if compute_covariance:
            raise NotImplementedError("Only the diagonal variances are supported for now.")

        source_covariance = self.source_covariance
        target_covariance = self.target_covariance

        if source_covariance is None or target_covariance is None:
            raise ValueError("Source and Target covariances must be set via set_covariances() for probabilistic forward pass.")

        source_activations = source_results.activations.to(self.device)
        target_activations = target_results.activations.to(self.device)

        # Add bias term to activations if necessary (controlled by class attribute)
        if self.source_projection_has_bias:
            source_activations = torch.cat([source_activations, torch.ones_like(source_activations[:, :1])], dim=-1)

        if self.target_projection_has_bias:
            target_activations = torch.cat([target_activations, torch.ones_like(target_activations[:, :1])], dim=-1)

        source_embeds = source_results.embeds.to(self.device)
        target_embeds = target_results.embeds.to(self.device)

        # Ensure covariances factors are on the correct device
        source_A_inv = source_covariance.A_inv # Already on device from set_covariances
        source_B_inv = source_covariance.B_inv
        target_A_inv = target_covariance.A_inv
        target_B_inv = target_covariance.B_inv

        source_B_factor = source_B_inv.diagonal()
        target_B_factor = target_B_inv.diagonal()

        # Ensure inputs to einsum are on the same device
        source_diag_cov = torch.einsum('ij,jk,ik->i', source_activations, source_A_inv, source_activations).unsqueeze(-1) * source_B_factor.unsqueeze(0)
        target_diag_cov = torch.einsum('ij,jk,ik->i', target_activations, target_A_inv, target_activations).unsqueeze(-1) * target_B_factor.unsqueeze(0)


        norm_source = source_embeds**2 + source_diag_cov
        expect_norm_source = norm_source.sum(dim=-1, keepdim=True)
        norm_target = target_embeds**2 + target_diag_cov
        expect_norm_target = norm_target.sum(dim=-1, keepdim=True)

        # Clamp norms to avoid division by zero or sqrt of negative numbers
        expect_norm_source = torch.clamp(expect_norm_source, min=1e-12)
        expect_norm_target = torch.clamp(expect_norm_target, min=1e-12)

        # compute expected value
        expected_similarity = torch.matmul(source_embeds / torch.sqrt(expect_norm_source), (target_embeds / torch.sqrt(expect_norm_target)).t())

        # compute variance
        term1 = torch.matmul(norm_source, target_diag_cov.t())
        term2 = torch.matmul(source_diag_cov, (target_embeds**2).t())

        variance_similarity = (term1 + term2) / (expect_norm_source @ expect_norm_target.t())

        # Clamp variance to be non-negative
        variance_similarity = torch.clamp(variance_similarity, min=0.0)

        scale = self.logit_scale.exp()

        logits_mean = expected_similarity * scale
        logits_var = variance_similarity * (scale**2) # Only diagonal variance calculated here

        if self.logit_bias is not None:
             logits_mean = logits_mean + self.logit_bias

        # Return diagonal variance [N_source, N_target]
        return ProbabilisticLogits(
            mean=logits_mean,
            var=logits_var,
        )

    def forward(
            self,
            source_results: EncoderResult,
            target_results: EncoderResult,
            map_estimate: bool = False,
        ):
        """
        Args:
            source_results (EncoderResult): Features for source items (e.g., images)
            target_results (EncoderResult): Features for target items (e.g., text prompts)
            map_estimate (bool): If True, perform deterministic forward pass.

        Returns:
            ProbabilisticLogits | torch.Tensor: Probabilistic or deterministic logits.
        """

        if map_estimate or self.source_covariance is None or self.target_covariance is None:
            # Perform deterministic calculation if requested or if covariances not set
            logits_map = self.deterministic_forward(source_results, target_results)
            if map_estimate: # Return ProbabilisticLogits with zero variance for consistency
                 covar_map = torch.zeros_like(logits_map) # Shape [N_source, N_target]
                 return ProbabilisticLogits(mean=logits_map, var=covar_map)
            else: # Return just the tensor if covariances weren't set but map_estimate=False
                 return logits_map

        # Perform probabilistic calculation
        return self._compute_probabilistic_logits_smith(source_results, target_results)

class SIGLIP(CLIP):
    source_projection_has_bias = True
    target_projection_has_bias = True

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        siglip = SiglipModel.from_pretrained(model_name)
        model = cls(
            logit_scale=siglip.logit_scale.item(),
            logit_bias=siglip.logit_bias.item(), # Siglip has a bias term
        )
        model = model.to(device) if device is not None else model
        return model

    # Inherits forward, deterministic_forward, _compute_probabilistic_logits_smith
    # The has_bias attributes control the behavior within _compute_probabilistic_logits_smith

# --- END OF FILE vlm.py ---