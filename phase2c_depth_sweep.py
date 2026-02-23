#!/usr/bin/env python3
"""Phase 2c: Aggressive Speculative Depth Sweep.

Single-seed experiment with three passes:
  Pass 1: Training + disk checkpoints (state_dict fp16 + optimizer momentum)
  Pass 2: K-sweep from disk (6 K values x 3 predictors x ~31 checkpoints)
  Pass 3: Cascaded workers at best stable checkpoints

Usage:
  python phase2c_depth_sweep.py --seed 42 [--max-steps 2000] [--skip-training] [--cleanup]
"""
import os, sys, time, json, gc, copy, math, argparse
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__) or '.')
from src.fingerprint.capture import build_probe_set, capture_at_checkpoint
from src.regime.detect import cosine_similarity_batch
from src.speculative.predictors import (
    apply_linear_prediction, apply_momentum_prediction,
    apply_momentum_prediction_from_state, apply_quadratic_prediction,
    restore_weights, compute_val_loss, compute_prediction_direction_norm,
)
from src.speculative.workers import (
    simulate_worker, check_worker_handoff, run_cascade,
    PredictionCache, compute_relative_l2,
)
from src.speculative.landscape import compute_landscape_probe_v2


# ============================================================
# Configuration
# ============================================================

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    max_steps: int = 2000
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 256
    checkpoint_interval: int = 50
    num_probes: int = 100
    probe_max_length: int = 128
    capture_layers: list = field(default_factory=lambda: ['final_hidden'])
    seed: int = 42
    gradient_accumulation_steps: int = 1
    device: str = "cpu"
    use_amp: bool = False  # Mixed precision (bf16 on A100)
    skip_inline_leap: bool = False  # Skip momentum K=5 leap in Pass 1 (saves GPU memory)


class SimpleTokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}


# Phase 2c sweep configuration
K_SWEEP_VALUES = [5, 10, 25, 50, 75, 100]
PREDICTORS = ['momentum', 'linear', 'quadratic']
SWEEP_REGIMES = ['stable', 'transition']
CHECKPOINT_INTERVAL = 50
MOMENTUM_SCALES = [0.8, 1.0, 1.2]

# Averaged thresholds from Phase 0 across 5 seeds
THRESHOLD_HIGH = np.mean([0.9998489, 0.9996176, 0.9994301, 0.9993165, 0.9992327])
THRESHOLD_LOW = np.mean([0.9974827, 0.9979310, 0.9953635, 0.9961008, 0.9944700])
CONSECUTIVE_STABLE_REQUIRED = 2

PHASE0_BASELINE_LOSS = 1.756
VAL_HOLDOUT = 200
VAL_BATCH_SIZE = 32
WORKER_HANDOFF_THRESHOLD = 0.05

# Cascade configurations: (chain_length, K_per_link)
CASCADE_CONFIGS = [
    (4, 25),   # 4 x K=25 = depth 100
    (2, 50),   # 2 x K=50 = depth 100
    (10, 10),  # 10 x K=10 = depth 100
]

# Landscape V2 fractions
LANDSCAPE_FRACTIONS = [0.25, 0.5, 1.0, 2.0]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.learning_rate * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))


def classify_regime(similarity):
    if similarity >= THRESHOLD_HIGH:
        return 'stable'
    elif similarity <= THRESHOLD_LOW:
        return 'chaotic'
    else:
        return 'transition'


def model_factory(cfg):
    """Create a GPT-2 model from config. Phase 3: swap to AutoModelForCausalLM."""
    return GPT2LMHeadModel(cfg)


# ============================================================
# Pass 1: Training + Disk Checkpoints
# ============================================================

def run_pass1_training(config, seed, output_dir, model_cfg, init_sd, train_ids, train_mask,
                       val_batch, canonical_probe_set):
    """Train the model, saving checkpoints to disk at every interval.

    Saves: state_dict (fp16), optimizer exp_avg/exp_avg_sq (fp16),
           step metadata, regime info.
    """
    print(f'\n{"=" * 60}')
    print(f'PASS 1: Training + Disk Checkpoints (seed {seed})')
    print(f'{"=" * 60}', flush=True)

    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = config.device
    set_seed(seed)
    model = model_factory(model_cfg)
    model.load_state_dict(init_sd)
    model.to(device)

    dataset = SimpleTokenDataset(train_ids, train_mask)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)

    # Move validation batch to device
    val_batch = {k: v.to(device) for k, v in val_batch.items()}

    # Mixed precision setup
    use_amp = config.use_amp and device != 'cpu'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    prev_activation = None
    consecutive_stable_count = 0
    step = 0
    total_steps_trained = 0
    total_steps_skipped = 0

    checkpoint_log = []
    prediction_log = []
    loss_curve_steps = []
    loss_curve_values = []
    val_loss_history = []

    prev_checkpoint_sd = None
    prev_prev_checkpoint_sd = None

    running_loss = 0.0
    loss_count = 0
    di = iter(dataloader)
    t0 = time.time()

    print(f'Training {config.max_steps} steps...', flush=True)

    while step < config.max_steps:
        try:
            batch = next(di)
        except StopIteration:
            di = iter(dataloader)
            batch = next(di)

        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                         labels=batch['input_ids'])
        scaler.scale(out.loss).backward()
        running_loss += out.loss.item()
        loss_count += 1

        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        step += 1
        total_steps_trained += 1

        if step % CHECKPOINT_INTERVAL == 0:
            avg_loss = running_loss / max(loss_count, 1)
            elapsed = time.time() - t0

            loss_curve_steps.append(step)
            loss_curve_values.append(avg_loss)

            # Regime detection
            model.eval()
            acts = capture_at_checkpoint(model, canonical_probe_set, ['final_hidden'], device=device)
            current_activation = acts['final_hidden']

            similarity = None
            regime = 'unknown'
            if prev_activation is not None:
                per_probe_sim = cosine_similarity_batch(prev_activation, current_activation)
                similarity = float(per_probe_sim.mean())
                regime = classify_regime(similarity)

            if regime == 'stable':
                consecutive_stable_count += 1
            else:
                consecutive_stable_count = 0

            is_stable_confirmed = consecutive_stable_count >= CONSECUTIVE_STABLE_REQUIRED

            # Get current state on device for predictions, then move to CPU for storage
            current_sd_device = {k: v.clone() for k, v in model.state_dict().items()}
            current_val_loss = compute_val_loss(model, val_batch)
            val_loss_history.append(current_val_loss)

            # Inline momentum K=5 evaluation for applying leaps
            # (skipped in memory-constrained mode — Pass 2 does full evaluation from disk)
            momentum_k5_improved = False
            if prev_checkpoint_sd is not None and not config.skip_inline_leap:
                backup = apply_momentum_prediction(model, optimizer, current_sd_device, 5, lr)
                pred_loss_k5 = compute_val_loss(model, val_batch)
                restore_weights(model, backup)
                momentum_k5_improved = pred_loss_k5 < current_val_loss

                prediction_log.append({
                    'step': step, 'K': 5, 'predictor': 'momentum',
                    'regime': regime, 'similarity': similarity,
                    'current_val_loss': current_val_loss,
                    'predicted_val_loss': pred_loss_k5,
                    'delta': pred_loss_k5 - current_val_loss,
                    'accepted_by_loss': momentum_k5_improved,
                    'actually_applied': False,
                })

                # Apply leap if stable confirmed and improved
                if is_stable_confirmed and momentum_k5_improved:
                    apply_momentum_prediction(model, optimizer, current_sd_device, 5, lr)
                    old_step = step
                    step += 5
                    total_steps_skipped += 5

                    new_lr = get_lr(step, config)
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr

                    # Zero optimizer momentum after jump
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            state = optimizer.state.get(p, {})
                            if 'exp_avg' in state:
                                state['exp_avg'].zero_()
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'].zero_()

                    prediction_log[-1]['actually_applied'] = True
                    print(f'  Step {old_step:5d} -> {step:5d} | LEAP momentum K=5 | '
                          f'Loss: {current_val_loss:.4f} -> {pred_loss_k5:.4f}', flush=True)

                    # Update current_sd after leap
                    current_sd_device = {k: v.clone() for k, v in model.state_dict().items()}

            # Move state dict to CPU for storage (frees GPU memory)
            current_sd = {k: v.cpu() for k, v in current_sd_device.items()}
            del current_sd_device
            if device != 'cpu':
                torch.cuda.empty_cache()

            # Save checkpoint to disk
            # Determine whether to save optimizer state (stable/transition only)
            save_optimizer = regime in ('stable', 'transition', 'unknown')

            ckpt_data = {
                'state_dict': {k: v.half() for k, v in current_sd.items()},
                'step': step,
                'val_loss': current_val_loss,
                'train_loss': avg_loss,
                'regime': regime,
                'similarity': similarity,
                'lr': lr,
                'consecutive_stable': consecutive_stable_count,
                'is_stable_confirmed': is_stable_confirmed,
                'momentum_k5_improved': momentum_k5_improved,
            }

            if save_optimizer:
                exp_avg_dict = {}
                exp_avg_sq_dict = {}
                for name, param in model.named_parameters():
                    state = optimizer.state.get(param, {})
                    if 'exp_avg' in state:
                        exp_avg_dict[name] = state['exp_avg'].cpu().half()
                    if 'exp_avg_sq' in state:
                        exp_avg_sq_dict[name] = state['exp_avg_sq'].cpu().half()
                ckpt_data['optimizer_exp_avg'] = exp_avg_dict
                ckpt_data['optimizer_exp_avg_sq'] = exp_avg_sq_dict

            ckpt_path = ckpt_dir / f'step_{step:05d}.pt'
            torch.save(ckpt_data, ckpt_path)

            checkpoint_log.append({
                'step': step,
                'train_loss': avg_loss,
                'val_loss': current_val_loss,
                'similarity': similarity,
                'regime': regime,
                'consecutive_stable': consecutive_stable_count,
                'is_stable_confirmed': is_stable_confirmed,
                'momentum_k5_improved': momentum_k5_improved,
                'elapsed': elapsed,
                'ckpt_path': str(ckpt_path),
            })

            if step % 200 == 0:
                print(f'  Step {step:5d} | Loss: {avg_loss:.4f} | Val: {current_val_loss:.4f} | '
                      f'LR: {lr:.2e} | Regime: {regime} | {elapsed:.0f}s', flush=True)

            prev_prev_checkpoint_sd = prev_checkpoint_sd
            prev_checkpoint_sd = current_sd
            prev_activation = current_activation
            running_loss = 0.0
            loss_count = 0

    total_time = time.time() - t0
    final_val_loss = compute_val_loss(model, val_batch)

    del model, optimizer
    gc.collect()

    pass1_results = {
        'seed': seed,
        'total_steps_trained': total_steps_trained,
        'total_steps_skipped': total_steps_skipped,
        'final_val_loss': final_val_loss,
        'total_time_seconds': total_time,
        'loss_curve': {'steps': loss_curve_steps, 'values': [float(v) for v in loss_curve_values]},
        'regime_timeline': {
            'steps': [c['step'] for c in checkpoint_log],
            'regimes': [c['regime'] for c in checkpoint_log],
            'similarities': [c['similarity'] for c in checkpoint_log],
        },
        'checkpoint_log': checkpoint_log,
        'prediction_log': prediction_log,
        'val_loss_history': [float(v) for v in val_loss_history],
    }

    results_path = output_dir / 'pass1_results.json'
    with open(results_path, 'w') as f:
        json.dump(pass1_results, f, indent=2)
    print(f'\nPass 1 done in {total_time:.0f}s. {len(checkpoint_log)} checkpoints saved.', flush=True)

    return pass1_results


# ============================================================
# Pass 2: K-Sweep from Disk
# ============================================================

def run_pass2_ksweep(config, seed, output_dir, model_cfg, val_batch):
    """Load checkpoints from disk, evaluate all predictors x K values.

    Three acceptance criteria logged in parallel:
      strict: predicted_loss < current_loss
      adaptive: predicted_loss <= current_loss + val_loss_stddev
      pct: predicted_loss <= current_loss * 1.005
    """
    print(f'\n{"=" * 60}')
    print(f'PASS 2: K-Sweep from Disk (seed {seed})')
    print(f'{"=" * 60}', flush=True)

    ckpt_dir = output_dir / 'checkpoints'
    ckpt_files = sorted(ckpt_dir.glob('step_*.pt'))
    print(f'Found {len(ckpt_files)} checkpoints', flush=True)

    # Load pass1 results for val_loss_history
    pass1_path = output_dir / 'pass1_results.json'
    with open(pass1_path) as f:
        pass1 = json.load(f)
    val_losses = pass1.get('val_loss_history', [])
    val_loss_std = float(np.std(val_losses)) if len(val_losses) > 1 else 0.001

    # Load checkpoints lazily to save memory
    num_ckpts = len(ckpt_files)

    evaluations = []
    landscape_probes = []
    t0 = time.time()

    device = config.device
    model = model_factory(model_cfg)
    model.to(device)
    val_batch = {k: v.to(device) for k, v in val_batch.items()}

    # Keep a sliding window of CPU-resident checkpoint data
    prev_ckpt = None
    prev_prev_ckpt = None

    for i, cp_file in enumerate(ckpt_files):
        ckpt = torch.load(cp_file, weights_only=False, map_location='cpu')
        step = ckpt['step']
        regime = ckpt['regime']
        current_loss = ckpt['val_loss']
        lr_val = ckpt['lr']

        # Skip chaotic checkpoints (always 0% acceptance)
        if regime == 'chaotic':
            prev_prev_ckpt = prev_ckpt
            prev_ckpt = ckpt
            continue

        # Keep current_sd on CPU — predictors handle .to(device) per-parameter
        current_sd = {k: v.float() for k, v in ckpt['state_dict'].items()}
        has_optimizer = 'optimizer_exp_avg' in ckpt
        has_prev = prev_ckpt is not None
        has_prev_prev = prev_prev_ckpt is not None

        # Load model with current checkpoint (moves to device via model.to)
        model.load_state_dict(current_sd, strict=True)
        # Note: model is already on device, so load_state_dict copies CPU->GPU

        # Keep all extra state dicts on CPU to save GPU memory
        prev_sd_cpu = {k: v.float() for k, v in prev_ckpt['state_dict'].items()} if has_prev else None
        prev_prev_sd_cpu = {k: v.float() for k, v in prev_prev_ckpt['state_dict'].items()} if has_prev_prev else None

        # Keep optimizer state on CPU — moved to device per-parameter in predictor
        exp_avg_cpu = None
        exp_avg_sq_cpu = None
        if has_optimizer:
            exp_avg_cpu = {k: v.float() for k, v in ckpt['optimizer_exp_avg'].items()}
            exp_avg_sq_cpu = {k: v.float() for k, v in ckpt['optimizer_exp_avg_sq'].items()}

        for K in K_SWEEP_VALUES:
            if step + K > config.max_steps:
                continue

            for predictor in PREDICTORS:
                predicted_loss = None

                if predictor == 'linear' and has_prev:
                    # Move prev_sd to device briefly
                    prev_sd = {k: v.to(device) for k, v in prev_sd_cpu.items()}
                    backup = apply_linear_prediction(model, prev_sd, current_sd, K, CHECKPOINT_INTERVAL)
                    predicted_loss = compute_val_loss(model, val_batch)
                    restore_weights(model, backup)
                    del prev_sd

                elif predictor == 'momentum' and has_optimizer:
                    best_loss = float('inf')
                    best_scale = 1.0
                    for sc in MOMENTUM_SCALES:
                        backup = apply_momentum_prediction_from_state(
                            model, current_sd, exp_avg_cpu, exp_avg_sq_cpu, K, lr_val, scale_factor=sc
                        )
                        pl = compute_val_loss(model, val_batch)
                        if pl < best_loss:
                            best_loss = pl
                            best_scale = sc
                        restore_weights(model, backup)

                    predicted_loss = best_loss

                elif predictor == 'quadratic' and has_prev and has_prev_prev:
                    prev_sd = {k: v.to(device) for k, v in prev_sd_cpu.items()}
                    prev_prev_sd = {k: v.to(device) for k, v in prev_prev_sd_cpu.items()}
                    backup = apply_quadratic_prediction(
                        model, prev_prev_sd, prev_sd, current_sd, K, CHECKPOINT_INTERVAL
                    )
                    predicted_loss = compute_val_loss(model, val_batch)
                    restore_weights(model, backup)
                    del prev_sd, prev_prev_sd

                if predicted_loss is not None:
                    accepted_strict = predicted_loss < current_loss
                    accepted_adaptive = predicted_loss <= current_loss + val_loss_std
                    accepted_pct = predicted_loss <= current_loss * 1.005

                    evaluations.append({
                        'step': step,
                        'K': K,
                        'predictor': predictor,
                        'regime': regime,
                        'current_loss': current_loss,
                        'predicted_loss': predicted_loss,
                        'delta': predicted_loss - current_loss,
                        'accepted_strict': accepted_strict,
                        'accepted_adaptive': accepted_adaptive,
                        'accepted_pct': accepted_pct,
                        'val_loss_stddev': val_loss_std,
                    })

            # Landscape probe V2 at this checkpoint (use momentum K=25 direction)
            if has_optimizer and step + 25 <= config.max_steps:
                backup = apply_momentum_prediction_from_state(
                    model, current_sd, exp_avg_cpu, exp_avg_sq_cpu, 25, lr_val
                )
                probe_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()}
                restore_weights(model, backup)

                probe_result = compute_landscape_probe_v2(
                    model, val_batch, probe_predicted_sd, current_sd,
                    fractions=LANDSCAPE_FRACTIONS,
                )
                probe_result['step'] = step
                probe_result['regime'] = regime
                landscape_probes.append(probe_result)

                del probe_predicted_sd

        # Free GPU memory for this checkpoint
        del current_sd, prev_sd_cpu, prev_prev_sd_cpu, exp_avg_cpu, exp_avg_sq_cpu
        if device != 'cpu':
            torch.cuda.empty_cache()

        # Slide the window
        prev_prev_ckpt = prev_ckpt
        prev_ckpt = ckpt

        if (i + 1) % 5 == 0:
            print(f'  Processed {i+1}/{num_ckpts} checkpoints '
                  f'({len(evaluations)} evaluations)...', flush=True)

    del model
    gc.collect()

    # Build summary_by_k
    summary_by_k = {}
    for K in K_SWEEP_VALUES:
        summary_by_k[K] = {}
        for predictor in PREDICTORS:
            summary_by_k[K][predictor] = {}
            for regime in SWEEP_REGIMES:
                entries = [e for e in evaluations
                           if e['K'] == K and e['predictor'] == predictor and e['regime'] == regime]
                n = len(entries)
                if n > 0:
                    summary_by_k[K][predictor][regime] = {
                        'n': n,
                        'strict_rate': sum(1 for e in entries if e['accepted_strict']) / n,
                        'adaptive_rate': sum(1 for e in entries if e['accepted_adaptive']) / n,
                        'pct_rate': sum(1 for e in entries if e['accepted_pct']) / n,
                    }

    elapsed = time.time() - t0
    pass2_results = {
        'k_values': K_SWEEP_VALUES,
        'predictors': PREDICTORS,
        'evaluations': evaluations,
        'summary_by_k': summary_by_k,
        'landscape_probe_v2': landscape_probes,
        'time_seconds': elapsed,
    }

    results_path = output_dir / 'pass2_results.json'
    with open(results_path, 'w') as f:
        json.dump(pass2_results, f, indent=2)

    # Print summary table
    print(f'\nK-Sweep Summary (strict acceptance):')
    print(f'{"K":>5} {"Predictor":<12} {"Regime":<12} {"Rate":>8} {"N":>5}')
    print('-' * 45)
    for K in K_SWEEP_VALUES:
        for pred in PREDICTORS:
            for regime in SWEEP_REGIMES:
                d = summary_by_k.get(K, {}).get(pred, {}).get(regime)
                if d:
                    print(f'{K:>5} {pred:<12} {regime:<12} {d["strict_rate"]:>7.1%} {d["n"]:>5}')

    print(f'\nPass 2 done in {elapsed:.0f}s. {len(evaluations)} evaluations.', flush=True)
    return pass2_results


# ============================================================
# Pass 3: Cascaded Workers
# ============================================================

def run_pass3_cascades(config, seed, output_dir, model_cfg, train_ids, train_mask,
                       val_batch):
    """Run cascaded worker configurations at best stable checkpoints."""
    print(f'\n{"=" * 60}')
    print(f'PASS 3: Cascaded Workers (seed {seed})')
    print(f'{"=" * 60}', flush=True)

    ckpt_dir = output_dir / 'checkpoints'

    # Load pass1 results to find best stable checkpoints
    with open(output_dir / 'pass1_results.json') as f:
        pass1 = json.load(f)

    # Find top 4 stable checkpoints by consecutive_stable count
    stable_ckpts = [c for c in pass1['checkpoint_log']
                    if c['regime'] == 'stable' and c['is_stable_confirmed']]
    stable_ckpts.sort(key=lambda c: c['consecutive_stable'], reverse=True)
    best_stable = stable_ckpts[:4]

    if not best_stable:
        print('No stable-confirmed checkpoints found. Skipping cascades.', flush=True)
        pass3_results = {'configurations': [], 'time_seconds': 0}
        results_path = output_dir / 'pass3_results.json'
        with open(results_path, 'w') as f:
            json.dump(pass3_results, f, indent=2)
        return pass3_results

    print(f'Using {len(best_stable)} stable checkpoints: '
          f'{[c["step"] for c in best_stable]}', flush=True)

    device = config.device
    dataset = SimpleTokenDataset(train_ids, train_mask)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(seed))
    di = iter(dataloader)
    val_batch = {k: v.to(device) for k, v in val_batch.items()}

    cascade_results = []
    t0 = time.time()

    for ckpt_meta in best_stable:
        step = ckpt_meta['step']
        ckpt_path = Path(ckpt_meta['ckpt_path'])
        if not ckpt_path.exists():
            print(f'  Checkpoint {ckpt_path} not found, skipping', flush=True)
            continue

        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        start_sd = {k: v.float().to(device) for k, v in ckpt['state_dict'].items()}
        lr_val = ckpt['lr']

        has_optimizer = 'optimizer_exp_avg' in ckpt
        if not has_optimizer:
            print(f'  Step {step}: no optimizer state, skipping', flush=True)
            continue

        exp_avg = {k: v.float().to(device) for k, v in ckpt['optimizer_exp_avg'].items()}
        exp_avg_sq = {k: v.float().to(device) for k, v in ckpt['optimizer_exp_avg_sq'].items()}

        def predict_fn(model, current_sd, K, lr):
            """Apply momentum prediction in-place."""
            eps = 1e-8
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in exp_avg and name in exp_avg_sq:
                        ea = exp_avg[name]
                        easq = exp_avg_sq[name]
                        update = ea / (torch.sqrt(easq) + eps)
                        param.data.copy_(current_sd[name] - K * lr * update)
                    elif name in current_sd:
                        param.data.copy_(current_sd[name])

        for chain_length, K_per_link in CASCADE_CONFIGS:
            total_depth = chain_length * K_per_link
            if step + total_depth > config.max_steps:
                continue

            print(f'  Step {step}: cascade {chain_length}x K={K_per_link} '
                  f'(depth {total_depth})...', flush=True)

            def lr_fn(s):
                return get_lr(s, config)

            result = run_cascade(
                model_factory=model_factory,
                model_cfg=model_cfg,
                start_sd=start_sd,
                chain_length=chain_length,
                K_per_link=K_per_link,
                data_iterator=di,
                dataloader=dataloader,
                val_batch=val_batch,
                lr_fn=lr_fn,
                max_steps=config.max_steps,
                current_step=step,
                predict_fn=predict_fn,
                device=device,
                use_amp=config.use_amp and device != 'cpu',
            )
            di = result.pop('data_iterator')

            cascade_results.append(result)

            if result['links']:
                drifts = [l['l2_drift'] for l in result['links']]
                print(f'    -> {len(result["links"])} links, '
                      f'L2 drift: {min(drifts):.6f} - {max(drifts):.6f}', flush=True)

        del ckpt, start_sd, exp_avg, exp_avg_sq
        gc.collect()

    elapsed = time.time() - t0
    pass3_results = {
        'configurations': cascade_results,
        'time_seconds': elapsed,
    }

    results_path = output_dir / 'pass3_results.json'
    with open(results_path, 'w') as f:
        json.dump(pass3_results, f, indent=2)

    print(f'\nPass 3 done in {elapsed:.0f}s. {len(cascade_results)} cascades.', flush=True)
    return pass3_results


# ============================================================
# Combine Results
# ============================================================

def combine_results(seed, output_dir):
    """Merge pass1/2/3 results into combined_results.json."""
    with open(output_dir / 'pass1_results.json') as f:
        pass1 = json.load(f)
    with open(output_dir / 'pass2_results.json') as f:
        pass2 = json.load(f)
    with open(output_dir / 'pass3_results.json') as f:
        pass3 = json.load(f)

    combined = {
        'config': {
            'seed': seed,
            'max_steps': pass1.get('total_steps_trained', 0) + pass1.get('total_steps_skipped', 0),
            'checkpoint_interval': CHECKPOINT_INTERVAL,
            'k_sweep_values': K_SWEEP_VALUES,
            'predictors': PREDICTORS,
            'threshold_high': float(THRESHOLD_HIGH),
            'threshold_low': float(THRESHOLD_LOW),
            'consecutive_stable_required': CONSECUTIVE_STABLE_REQUIRED,
            'cascade_configs': CASCADE_CONFIGS,
            'landscape_fractions': LANDSCAPE_FRACTIONS,
            'phase0_baseline_loss': PHASE0_BASELINE_LOSS,
        },
        'summary': {
            'total_steps_trained': pass1['total_steps_trained'],
            'total_steps_skipped': pass1['total_steps_skipped'],
            'final_val_loss': pass1['final_val_loss'],
            'pass1_time': pass1['total_time_seconds'],
            'pass2_time': pass2['time_seconds'],
            'pass3_time': pass3['time_seconds'],
        },
        'loss_curve': pass1['loss_curve'],
        'regime_timeline': pass1['regime_timeline'],
        'prediction_log': pass1['prediction_log'],
        'depth_sweep': {
            'k_values': pass2['k_values'],
            'evaluations': pass2['evaluations'],
            'summary_by_k': pass2['summary_by_k'],
        },
        'cascade_results': pass3,
        'landscape_probe_v2': pass2['landscape_probe_v2'],
    }

    results_path = output_dir / 'combined_results.json'
    with open(results_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f'Combined results saved to {results_path}', flush=True)
    return combined


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 2c: Aggressive Speculative Depth Sweep')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip Pass 1 (use existing checkpoints)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Delete checkpoints after aggregation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use mixed precision (bf16 on A100)')
    args = parser.parse_args()

    seed = args.seed
    output_dir = Path(f'results/phase2c/seed{seed}')
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(max_steps=args.max_steps, seed=seed,
                            device=args.device, use_amp=args.use_amp)

    print(f'Phase 2c: Aggressive Speculative Depth Sweep')
    print(f'Seed: {seed}, Max steps: {config.max_steps}')
    print(f'K sweep: {K_SWEEP_VALUES}')
    print(f'Cascade configs: {CASCADE_CONFIGS}')
    print(f'Output: {output_dir}', flush=True)

    # Load shared resources
    print('\nLoading pre-tokenized data...')
    cached = torch.load('/tmp/phase0_tokenized.pt', weights_only=True)
    all_ids = cached['input_ids']
    all_mask = cached['attention_mask']
    del cached; gc.collect()

    train_ids = all_ids[:-VAL_HOLDOUT]
    train_mask = all_mask[:-VAL_HOLDOUT]
    val_ids = all_ids[-VAL_HOLDOUT:]
    val_mask = all_mask[-VAL_HOLDOUT:]

    val_batch = {
        'input_ids': val_ids[:VAL_BATCH_SIZE],
        'attention_mask': val_mask[:VAL_BATCH_SIZE],
    }

    print('Loading model config...')
    model_cfg = AutoConfig.from_pretrained('/tmp/gpt2_local')
    init_sd = torch.load('/tmp/gpt2_local/model.pt', weights_only=True, map_location='cpu')

    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token

    canonical_probe_set = build_probe_set(tok, num_probes=100, max_length=128, seed=0)
    print('Ready.\n', flush=True)

    # Pass 1: Training
    if not args.skip_training:
        run_pass1_training(config, seed, output_dir, model_cfg, init_sd,
                          train_ids, train_mask, val_batch, canonical_probe_set)
    else:
        print('Skipping Pass 1 (--skip-training)', flush=True)

    del init_sd; gc.collect()

    # Pass 2: K-sweep
    run_pass2_ksweep(config, seed, output_dir, model_cfg, val_batch)

    # Pass 3: Cascades
    run_pass3_cascades(config, seed, output_dir, model_cfg, train_ids, train_mask, val_batch)

    # Combine
    combined = combine_results(seed, output_dir)

    # Cleanup checkpoints if requested
    if args.cleanup:
        import shutil
        ckpt_dir = output_dir / 'checkpoints'
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f'Cleaned up {ckpt_dir}')

    # Generate plots
    print('\nGenerating plots...')
    try:
        from src.analysis.plot_phase2c import plot_single_seed
        import matplotlib
        matplotlib.use('Agg')
        plot_single_seed(combined, save_dir=str(output_dir))
        print('Plots saved.')
    except Exception as e:
        print(f'Plot generation failed: {e}')
        import traceback
        traceback.print_exc()

    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
