"""
Activation capture for probe inputs.

Runs a fixed set of probe inputs through a model and extracts activation
vectors at specified layers using forward hooks. These activation snapshots
are the raw signal for regime detection.
"""

import torch
import numpy as np
from typing import Optional


class ActivationCapture:
    """Captures activation vectors from specified model layers during forward pass."""

    def __init__(self, model, layer_names: list[str]):
        """
        Args:
            model: HuggingFace model (or any nn.Module)
            layer_names: List of layer names to capture. Use dot notation
                         for nested modules (e.g., 'transformer.h.11.mlp').
                         Special name 'final_hidden' captures the last hidden
                         state from the model output.
        """
        self.model = model
        self.layer_names = layer_names
        self._activations = {}
        self._hooks = []

    def _get_module(self, name: str) -> torch.nn.Module:
        """Resolve dot-notation name to actual module."""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # Handle tuple outputs (common in transformers)
            if isinstance(output, tuple):
                output = output[0]
            # Store mean-pooled activation (average over sequence length)
            # Shape: (batch_size, hidden_dim)
            if output.dim() == 3:
                self._activations[name] = output.mean(dim=1).detach().cpu()
            else:
                self._activations[name] = output.detach().cpu()
        return hook_fn

    def register_hooks(self):
        """Register forward hooks on all target layers."""
        self.remove_hooks()
        for name in self.layer_names:
            if name == 'final_hidden':
                continue  # Handled separately from model output
            module = self._get_module(name)
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._activations = {}

    @torch.no_grad()
    def capture(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Run inputs through the model and return captured activations.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Optional attention mask

        Returns:
            Dict mapping layer_name -> activation tensor (batch_size, hidden_dim)
        """
        self._activations = {}
        self.model.eval()

        kwargs = {'input_ids': input_ids}
        if attention_mask is not None:
            kwargs['attention_mask'] = attention_mask

        output = self.model(**kwargs)

        return dict(self._activations)


def build_probe_set(tokenizer, num_probes: int = 100, max_length: int = 128,
                    seed: int = 42) -> dict:
    """
    Build a fixed probe set from diverse text prompts.

    Returns a dict with 'input_ids' and 'attention_mask' tensors ready
    for batched inference.
    """
    rng = np.random.RandomState(seed)

    # Diverse probe categories to cover different functional regions
    probe_templates = [
        # Factual
        "The capital of {country} is",
        "In the year {year}, the most significant event was",
        "The chemical formula for {compound} is",
        # Reasoning
        "If all cats are animals and all animals breathe, then cats",
        "The number that comes after {n} is",
        "To solve {n} + {m}, you compute",
        # Language
        "The opposite of '{word}' is",
        "Translate '{phrase}' to French:",
        "The plural of '{word}' is",
        # Code
        "def fibonacci(n):\n    ",
        "import numpy as np\nnp.array([1, 2, 3]).",
        "for i in range(10):\n    print(",
        # Creative
        "Once upon a time, in a land far away,",
        "The sunset painted the sky in shades of",
        "She opened the door and found",
        # Technical
        "The gradient of the loss function with respect to",
        "In quantum mechanics, the wave function",
        "The Big O complexity of binary search is",
        # Conversational
        "Hello, how are you today?",
        "Can you explain what machine learning is?",
        "What do you think about",
    ]

    countries = ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Germany", "Australia"]
    years = ["1969", "1776", "2000", "1945", "1989", "2020", "1066", "1492"]
    compounds = ["water", "salt", "glucose", "methane", "ethanol", "carbon dioxide"]
    words = ["happy", "fast", "bright", "cold", "large", "quiet", "simple", "strong"]
    phrases = ["good morning", "thank you", "how are you", "goodbye"]
    numbers = list(range(1, 50))

    probes = []
    for i in range(num_probes):
        template = probe_templates[i % len(probe_templates)]
        text = template
        if '{country}' in text:
            text = text.replace('{country}', rng.choice(countries))
        if '{year}' in text:
            text = text.replace('{year}', rng.choice(years))
        if '{compound}' in text:
            text = text.replace('{compound}', rng.choice(compounds))
        if '{word}' in text:
            text = text.replace('{word}', rng.choice(words))
        if '{phrase}' in text:
            text = text.replace('{phrase}', rng.choice(phrases))
        if '{n}' in text:
            n = int(rng.choice(numbers))
            text = text.replace('{n}', str(n))
        if '{m}' in text:
            m = int(rng.choice(numbers))
            text = text.replace('{m}', str(m))
        probes.append(text)

    # Tokenize all probes
    encoded = tokenizer(
        probes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'texts': probes,
    }


def _resolve_final_hidden(model) -> str:
    """
    Find the actual module name for the final hidden layer.

    For GPT-2: transformer.ln_f (final layer norm before lm_head)
    For other HF models: tries common patterns.
    """
    # GPT-2 style
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return 'transformer.ln_f'
    # GPT-NeoX / Pythia style
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
        return 'gpt_neox.final_layer_norm'
    # LLaMA / Mistral style
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return 'model.norm'
    # Fallback: try to find any final norm layer
    last_name = None
    for name, _ in model.named_modules():
        if 'ln_f' in name or 'final_layer_norm' in name or name.endswith('.norm'):
            last_name = name
    if last_name:
        return last_name
    raise ValueError(f"Cannot find final hidden layer for {type(model).__name__}. "
                     f"Specify layer name explicitly instead of 'final_hidden'.")


def capture_at_checkpoint(model, probe_set: dict, layer_names: list[str],
                          device: str = 'cuda', batch_size: int = 32) -> dict[str, np.ndarray]:
    """
    Capture activations for all probes at a single checkpoint.

    Args:
        model: The model at current checkpoint state
        probe_set: Output of build_probe_set()
        layer_names: Layers to capture. Use 'final_hidden' to auto-detect
                     the final layer norm.
        device: Device for inference
        batch_size: Batch size for probe inference

    Returns:
        Dict mapping layer_name -> numpy array of shape (num_probes, hidden_dim)
    """
    # Resolve 'final_hidden' to actual module name
    resolved_names = []
    name_map = {}  # resolved -> original
    for name in layer_names:
        if name == 'final_hidden':
            actual = _resolve_final_hidden(model)
            resolved_names.append(actual)
            name_map[actual] = 'final_hidden'
        else:
            resolved_names.append(name)
            name_map[name] = name

    capturer = ActivationCapture(model, resolved_names)
    capturer.register_hooks()

    input_ids = probe_set['input_ids']
    attention_mask = probe_set['attention_mask']
    num_probes = input_ids.shape[0]

    all_activations = {orig: [] for orig in layer_names}

    for start in range(0, num_probes, batch_size):
        end = min(start + batch_size, num_probes)
        batch_ids = input_ids[start:end].to(device)
        batch_mask = attention_mask[start:end].to(device)

        acts = capturer.capture(batch_ids, batch_mask)
        # Map resolved names back to original names
        for resolved, orig in name_map.items():
            if resolved in acts:
                all_activations[orig].append(acts[resolved].numpy())

    capturer.remove_hooks()

    # Concatenate batches
    result = {}
    for name in layer_names:
        if all_activations[name]:
            result[name] = np.concatenate(all_activations[name], axis=0)

    return result
