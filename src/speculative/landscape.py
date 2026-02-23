"""
Landscape probing V2 with adaptive epsilon.

Key improvements over Phase 2b:
1. epsilon = norm(predicted - current) — actual prediction magnitude, not fixed 1e-3
2. Evaluates loss at [0.25, 0.5, 1.0, 2.0]x the prediction direction
3. Curvature estimate from 3-point finite difference
4. 4 forward passes per checkpoint (vs 2 in Phase 2b)

Phase 3 ready: model-agnostic, operates on state dicts.
"""

import torch
import math


def compute_landscape_probe_v2(model, val_batch, predicted_sd, current_sd,
                                fractions=(0.25, 0.5, 1.0, 2.0)):
    """Compute loss landscape along prediction direction at multiple scales.

    Args:
        model: Current model (will be temporarily modified and restored)
        val_batch: Validation batch for loss computation
        predicted_sd: State dict of predicted weights
        current_sd: Current state dict
        fractions: Scale factors along prediction direction to evaluate

    Returns:
        dict with:
            direction_norm: L2 norm of prediction direction
            fractions: list of scale factors evaluated
            losses_at_fractions: loss at each fraction
            sensitivities_at_fractions: |loss - loss_at_1.0| for each fraction
            directional_derivative: finite difference approximation at scale 1.0
            curvature_estimate: second derivative from 3-point stencil
    """
    from .predictors import compute_val_loss

    # Compute direction: predicted - current (on model's device)
    device = next(model.parameters()).device
    direction = {}
    sum_sq = 0.0
    for name in current_sd:
        if name in predicted_sd:
            d = predicted_sd[name].float().to(device) - current_sd[name].float().to(device)
            direction[name] = d
            sum_sq += torch.sum(d ** 2).item()

    direction_norm = math.sqrt(sum_sq)

    if direction_norm < 1e-12:
        return {
            'direction_norm': 0.0,
            'fractions': list(fractions),
            'losses_at_fractions': [0.0] * len(fractions),
            'sensitivities_at_fractions': [0.0] * len(fractions),
            'directional_derivative': 0.0,
            'curvature_estimate': 0.0,
        }

    # Save original state
    backup = {k: v.clone() for k, v in model.state_dict().items()}

    # Loss at current point (fraction=0)
    loss_at_0 = compute_val_loss(model, val_batch)

    # Evaluate at each fraction
    losses = []
    for frac in fractions:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in direction:
                    param.data.copy_(current_sd[name].to(device) + frac * direction[name])

        loss = compute_val_loss(model, val_batch)
        losses.append(loss)

        # Restore for next iteration
        model.load_state_dict(backup)

    # Find loss at fraction=1.0 for sensitivity calculation
    idx_1 = list(fractions).index(1.0) if 1.0 in fractions else -1
    loss_at_1 = losses[idx_1] if idx_1 >= 0 else losses[-1]

    sensitivities = [abs(l - loss_at_1) for l in losses]

    # Directional derivative: (L(current + direction) - L(current)) / ||direction||
    directional_derivative = (loss_at_1 - loss_at_0) / direction_norm

    # Curvature estimate from 3-point finite difference
    # Need losses at 0, 0.5, 1.0 (or closest available fractions)
    curvature_estimate = 0.0
    frac_list = list(fractions)
    if 0.5 in frac_list and 1.0 in frac_list:
        idx_05 = frac_list.index(0.5)
        idx_10 = frac_list.index(1.0)
        h = 0.5 * direction_norm  # step size in parameter space
        if h > 1e-12:
            # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
            # Here: f(0), f(0.5*d), f(1.0*d) with h = 0.5*||d||
            curvature_estimate = (losses[idx_10] - 2 * losses[idx_05] + loss_at_0) / (h * h)
    elif len(losses) >= 3:
        # Fallback: use first three fractions
        h = (frac_list[1] - frac_list[0]) * direction_norm
        if h > 1e-12:
            curvature_estimate = (losses[2] - 2 * losses[1] + losses[0]) / (h * h)

    return {
        'direction_norm': direction_norm,
        'fractions': list(fractions),
        'losses_at_fractions': losses,
        'loss_at_origin': loss_at_0,
        'sensitivities_at_fractions': sensitivities,
        'directional_derivative': directional_derivative,
        'curvature_estimate': curvature_estimate,
    }
