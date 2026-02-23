"""
Extracted predictor functions for speculative weight prediction.

Three predictors operating on state dicts:
1. Linear — extrapolates from two consecutive checkpoints
2. Momentum — uses AdamW exp_avg/exp_avg_sq for K-step prediction
3. Quadratic — fits parabola to three consecutive checkpoints

All predictors modify the model in-place and return a backup for rollback.
Phase 3 ready: operates on state dicts, no model-specific assumptions.
"""

import torch
import math


def apply_linear_prediction(model, prev_sd, current_sd, K, interval):
    """Linear weight extrapolation: w_pred = w_current + (K/interval) * delta.

    Args:
        model: Model to modify in-place
        prev_sd: Previous checkpoint state_dict
        current_sd: Current checkpoint state_dict
        K: Steps to predict ahead
        interval: Steps between checkpoints

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    scale = K / interval
    device = next(model.parameters()).device

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in prev_sd and name in current_sd:
                cur = current_sd[name].to(device)
                prev = prev_sd[name].to(device)
                param.data.copy_(cur + scale * (cur - prev))

    return backup_sd


def apply_momentum_prediction(model, optimizer, current_sd, K, lr, scale_factor=1.0):
    """Momentum-informed prediction using AdamW exp_avg/exp_avg_sq.

    w_pred = w_current - scale * K * lr * exp_avg / (sqrt(exp_avg_sq) + eps)

    This IS the Adam update direction, extrapolated K steps.

    Args:
        model: Model to modify in-place
        optimizer: AdamW optimizer with populated state
        current_sd: Current checkpoint state_dict
        K: Steps to predict ahead
        lr: Current learning rate
        scale_factor: Multiplier for extrapolation (0.8-1.2)

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    eps = 1e-8

    with torch.no_grad():
        for name, param in model.named_parameters():
            state = optimizer.state.get(param, {})
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                update = exp_avg / (torch.sqrt(exp_avg_sq) + eps)
                param.data.copy_(current_sd[name] - scale_factor * K * lr * update)
            elif name in current_sd:
                param.data.copy_(current_sd[name])

    return backup_sd


def apply_momentum_prediction_from_state(model, current_sd, exp_avg_dict, exp_avg_sq_dict,
                                          K, lr, scale_factor=1.0):
    """Momentum prediction from saved optimizer state dicts (for Pass 2 disk replay).

    Same math as apply_momentum_prediction but takes exp_avg/exp_avg_sq as
    name->tensor dicts instead of requiring a live optimizer.

    Args:
        model: Model to modify in-place
        current_sd: Current checkpoint state_dict
        exp_avg_dict: {param_name: exp_avg_tensor}
        exp_avg_sq_dict: {param_name: exp_avg_sq_tensor}
        K: Steps to predict ahead
        lr: Current learning rate
        scale_factor: Multiplier for extrapolation

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    eps = 1e-8

    device = next(model.parameters()).device
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in exp_avg_dict and name in exp_avg_sq_dict:
                ea = exp_avg_dict[name].float().to(device)
                easq = exp_avg_sq_dict[name].float().to(device)
                update = ea / (torch.sqrt(easq) + eps)
                param.data.copy_(current_sd[name].to(device) - scale_factor * K * lr * update)
            elif name in current_sd:
                param.data.copy_(current_sd[name].to(device))

    return backup_sd


def apply_quadratic_prediction(model, sd_t0, sd_t1, sd_t2, K, interval):
    """Quadratic prediction using 3 consecutive checkpoints.

    Fits a parabola through the 3 weight snapshots:
      v1 = sd_t1 - sd_t0    (velocity at interval 1)
      v2 = sd_t2 - sd_t1    (velocity at interval 2)
      accel = v2 - v1        (acceleration)
      w_pred = sd_t2 + (K/interval) * v2 + 0.5 * (K/interval)^2 * accel

    Acceleration clamped to 0.5x velocity magnitude.

    Args:
        model: Model to modify in-place
        sd_t0, sd_t1, sd_t2: Three consecutive checkpoint state_dicts
        K: Steps to predict ahead
        interval: Steps between checkpoints

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    t = K / interval
    device = next(model.parameters()).device

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in sd_t0 and name in sd_t1 and name in sd_t2:
                t0 = sd_t0[name].to(device)
                t1 = sd_t1[name].to(device)
                t2 = sd_t2[name].to(device)
                v1 = t1 - t0
                v2 = t2 - t1
                accel = v2 - v1

                v2_norm = torch.norm(v2)
                accel_norm = torch.norm(accel)
                if accel_norm > 0 and v2_norm > 0:
                    max_accel = 0.5 * v2_norm
                    if accel_norm > max_accel:
                        accel = accel * (max_accel / accel_norm)

                pred = t2 + t * v2 + 0.5 * t * t * accel
                param.data.copy_(pred)

    return backup_sd


def restore_weights(model, backup_sd):
    """Restore model weights from backup."""
    model.load_state_dict(backup_sd)


def compute_val_loss(model, val_batch):
    """Single forward pass on fixed validation batch. Returns scalar loss.

    Automatically uses bf16 autocast when model is on CUDA.
    """
    model.eval()
    device = next(model.parameters()).device
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
        out = model(
            input_ids=val_batch['input_ids'],
            attention_mask=val_batch['attention_mask'],
            labels=val_batch['input_ids']
        )
    return out.loss.item()


def compute_prediction_direction_norm(predicted_sd, current_sd):
    """Compute L2 norm of the prediction direction (predicted - current).

    Returns:
        float: L2 norm across all parameters
    """
    sum_sq = 0.0
    for name in current_sd:
        if name in predicted_sd:
            diff = predicted_sd[name].float() - current_sd[name].float()
            sum_sq += torch.sum(diff ** 2).item()
    return math.sqrt(sum_sq)
