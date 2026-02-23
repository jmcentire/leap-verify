"""
Worker simulation and cascade pipeline for ASC architecture.

Provides:
1. simulate_worker() — run a speculative worker ahead from predicted state
2. check_worker_handoff() — verify predicted start matches actual state
3. PredictionCache — stores (start, end) pairs for fast-forwarding
4. run_cascade() — chain multiple workers for deep speculation (K=25x4, etc.)

Phase 3 ready: model_factory parameter swaps GPT2LMHeadModel for any model.
"""

import gc
import math
import torch
from dataclasses import dataclass, field


def simulate_worker(model_factory, model_cfg, init_sd, worker_config, data_iterator,
                    dataloader, val_batch, lr_fn, max_steps, current_step, device='cpu',
                    use_amp=False):
    """Simulate a speculative worker running ahead from a predicted state.

    Args:
        model_factory: Callable(config) -> model (e.g., GPT2LMHeadModel)
        model_cfg: Model config for creating temp model
        init_sd: State dict to start from (predicted weights)
        worker_config: (predictor_type, K) tuple
        data_iterator: Current data iterator position
        dataloader: Full dataloader for re-iteration
        val_batch: Validation batch for loss eval
        lr_fn: Callable(step) -> learning_rate
        max_steps: Maximum training step
        current_step: Current training step
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        dict with worker results
    """
    from .predictors import compute_val_loss

    predictor_type, K = worker_config

    temp_model = model_factory(model_cfg)
    temp_model.load_state_dict(init_sd)
    temp_model.to(device)

    temp_optimizer = torch.optim.AdamW(temp_model.parameters(), lr=lr_fn(current_step),
                                        weight_decay=0.01)

    di = data_iterator
    steps_done = 0
    for _ in range(K):
        sim_step = current_step + steps_done
        if sim_step >= max_steps:
            break

        try:
            batch = next(di)
        except StopIteration:
            di = iter(dataloader)
            batch = next(di)

        temp_model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            out = temp_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                             labels=batch['input_ids'])
        out.loss.backward()

        lr = lr_fn(sim_step)
        for pg in temp_optimizer.param_groups:
            pg['lr'] = lr
        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), 1.0)
        temp_optimizer.step()
        temp_optimizer.zero_grad()
        steps_done += 1

    end_loss = compute_val_loss(temp_model, val_batch)
    end_sd = {k: v.clone() for k, v in temp_model.state_dict().items()}
    start_sd_fp16 = {k: v.half() for k, v in init_sd.items()}

    del temp_model, temp_optimizer
    gc.collect()

    return {
        'predictor_type': predictor_type,
        'K': K,
        'steps_trained': steps_done,
        'end_loss': end_loss,
        'end_sd': end_sd,
        'start_sd_fp16': start_sd_fp16,
        'data_iterator': di,
    }


def check_worker_handoff(worker_result, actual_sd, threshold=0.05):
    """Check if a worker's predicted start is close enough to actual state.

    Args:
        worker_result: Result from simulate_worker (must have start_sd_fp16)
        actual_sd: Actual model state_dict at this checkpoint
        threshold: Relative L2 distance threshold

    Returns:
        (accepted, relative_l2_distance)
    """
    predicted_start = worker_result['start_sd_fp16']

    sum_sq_diff = 0.0
    sum_sq_actual = 0.0

    for name in actual_sd:
        if name in predicted_start:
            actual = actual_sd[name].float()
            predicted = predicted_start[name].float()
            sum_sq_diff += torch.sum((actual - predicted) ** 2).item()
            sum_sq_actual += torch.sum(actual ** 2).item()

    if sum_sq_actual < 1e-12:
        return False, float('inf')

    relative_l2 = math.sqrt(sum_sq_diff / sum_sq_actual)
    accepted = relative_l2 < threshold
    return accepted, relative_l2


def compute_relative_l2(sd_a, sd_b):
    """Compute relative L2 distance between two state dicts.

    Returns:
        float: relative L2 = ||a - b|| / ||a||
    """
    sum_sq_diff = 0.0
    sum_sq_a = 0.0

    for name in sd_a:
        if name in sd_b:
            a = sd_a[name].float()
            b = sd_b[name].float()
            sum_sq_diff += torch.sum((a - b) ** 2).item()
            sum_sq_a += torch.sum(a ** 2).item()

    if sum_sq_a < 1e-12:
        return float('inf')
    return math.sqrt(sum_sq_diff / sum_sq_a)


def run_cascade(model_factory, model_cfg, start_sd, chain_length, K_per_link,
                data_iterator, dataloader, val_batch, lr_fn, max_steps,
                current_step, predict_fn, device='cpu', use_amp=False):
    """Run a cascaded speculation pipeline: chain multiple workers end-to-end.

    Each link: predict K steps ahead from current state, train K steps from
    predicted state, pass end state to next link. Measures L2 drift per link.

    Args:
        model_factory: Callable(config) -> model
        model_cfg: Model config
        start_sd: Starting state dict (actual trained checkpoint)
        chain_length: Number of links in the cascade
        K_per_link: Steps per link
        data_iterator: Current data iterator
        dataloader: Full dataloader
        val_batch: Validation batch
        lr_fn: Callable(step) -> lr
        max_steps: Maximum training step
        current_step: Current training step at cascade start
        predict_fn: Callable(model, current_sd, K, lr) -> None
            Applies prediction to model in-place (e.g., momentum prediction).
            Should modify model weights to predicted state.
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        dict with cascade results including per-link metrics
    """
    from .predictors import compute_val_loss

    links = []
    current_link_sd = {k: v.clone() for k, v in start_sd.items()}
    di = data_iterator
    sim_step = current_step

    for link_idx in range(chain_length):
        if sim_step + K_per_link > max_steps:
            break

        # Create model, load current state, apply prediction
        temp_model = model_factory(model_cfg)
        temp_model.load_state_dict(current_link_sd)
        temp_model.to(device)

        # Get LR at this point
        link_lr = lr_fn(sim_step)

        # Apply prediction to get predicted state
        predict_fn(temp_model, current_link_sd, K_per_link, link_lr)
        predicted_sd = {k: v.clone() for k, v in temp_model.state_dict().items()}

        # Compute predicted loss
        predicted_loss = compute_val_loss(temp_model, val_batch)

        # Now train K steps from the predicted state
        temp_optimizer = torch.optim.AdamW(temp_model.parameters(), lr=link_lr,
                                            weight_decay=0.01)

        steps_done = 0
        for _ in range(K_per_link):
            if sim_step + steps_done >= max_steps:
                break
            try:
                batch = next(di)
            except StopIteration:
                di = iter(dataloader)
                batch = next(di)

            temp_model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                out = temp_model(input_ids=batch['input_ids'],
                                 attention_mask=batch['attention_mask'],
                                 labels=batch['input_ids'])
            out.loss.backward()

            lr = lr_fn(sim_step + steps_done)
            for pg in temp_optimizer.param_groups:
                pg['lr'] = lr
            torch.nn.utils.clip_grad_norm_(temp_model.parameters(), 1.0)
            temp_optimizer.step()
            temp_optimizer.zero_grad()
            steps_done += 1

        # End-of-link state
        end_loss = compute_val_loss(temp_model, val_batch)
        end_sd = {k: v.clone() for k, v in temp_model.state_dict().items()}

        # L2 drift: distance from predicted start to actual end
        l2_drift = compute_relative_l2(predicted_sd, end_sd)

        links.append({
            'link': link_idx,
            'start_step': sim_step,
            'K': K_per_link,
            'steps_trained': steps_done,
            'predicted_loss': predicted_loss,
            'trained_loss': end_loss,
            'l2_drift': l2_drift,
        })

        # Pass end state to next link
        current_link_sd = end_sd
        sim_step += steps_done

        del temp_model, temp_optimizer
        gc.collect()

    return {
        'chain_length': chain_length,
        'K_per_link': K_per_link,
        'checkpoint_step': current_step,
        'total_depth': sum(l['steps_trained'] for l in links),
        'links': links,
        'data_iterator': di,
    }


class PredictionCache:
    """Cache of successful (start_state, end_state) pairs for fast-forwarding."""

    def __init__(self, max_entries=5, hit_threshold=0.03):
        self.max_entries = max_entries
        self.hit_threshold = hit_threshold
        self.entries = []
        self.event_log = []

    def insert(self, start_sd, end_sd, metadata):
        """Insert a successful prediction pair."""
        start_fp16 = {k: v.half() for k, v in start_sd.items()}
        entry = (start_fp16, {k: v.clone() for k, v in end_sd.items()}, metadata)
        self.entries.append(entry)

        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

        self.event_log.append({
            'type': 'insert',
            'step': metadata.get('step', -1),
            'predictor': metadata.get('predictor', 'unknown'),
            'K': metadata.get('K', 0),
            'num_entries': len(self.entries),
        })

    def lookup(self, current_sd, step):
        """Look up cache for a state close to current_sd.

        Returns:
            (hit, end_sd, metadata) if found, else (False, None, None)
        """
        for start_fp16, end_sd, metadata in self.entries:
            sum_sq_diff = 0.0
            sum_sq_current = 0.0

            for name in current_sd:
                if name in start_fp16:
                    curr = current_sd[name].float()
                    cached = start_fp16[name].float()
                    sum_sq_diff += torch.sum((curr - cached) ** 2).item()
                    sum_sq_current += torch.sum(curr ** 2).item()

            if sum_sq_current < 1e-12:
                continue

            relative_l2 = math.sqrt(sum_sq_diff / sum_sq_current)

            if relative_l2 < self.hit_threshold:
                self.event_log.append({
                    'type': 'hit',
                    'step': step,
                    'distance': relative_l2,
                    'cached_step': metadata.get('step', -1),
                    'cached_K': metadata.get('K', 0),
                })
                return True, end_sd, metadata

        self.event_log.append({
            'type': 'miss',
            'step': step,
            'num_entries': len(self.entries),
        })
        return False, None, None
