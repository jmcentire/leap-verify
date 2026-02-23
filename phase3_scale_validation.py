#!/usr/bin/env python3
"""Phase 3: Scale Validation — Replicate Phase 2c on larger models.

Thin adapter over phase2c_depth_sweep.py. Overrides:
1. Model factory (AutoModelForCausalLM instead of GPT2LMHeadModel)
2. Tokenizer and data loading (model-specific tokenizer)
3. Checkpoint strategy (rolling window to manage disk)
4. Config (batch_size, model_name)

Usage:
  python phase3_scale_validation.py --seed 42 --model Qwen/Qwen2.5-1.5B [--max-steps 2000]
  python phase3_scale_validation.py --seed 42 --model microsoft/phi-2 --batch-size 2
"""
import os, sys, time, json, gc, argparse, shutil
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__) or '.')

# Import everything from phase2c except model-specific bits
from phase2c_depth_sweep import (
    TrainingConfig, SimpleTokenDataset, set_seed, get_lr, classify_regime,
    K_SWEEP_VALUES, PREDICTORS, SWEEP_REGIMES, CHECKPOINT_INTERVAL,
    MOMENTUM_SCALES, THRESHOLD_HIGH, THRESHOLD_LOW, CONSECUTIVE_STABLE_REQUIRED,
    VAL_HOLDOUT, VAL_BATCH_SIZE, WORKER_HANDOFF_THRESHOLD,
    CASCADE_CONFIGS, LANDSCAPE_FRACTIONS,
    run_pass1_training, run_pass2_ksweep, run_pass3_cascades, combine_results,
)
from src.fingerprint.capture import build_probe_set

# ============================================================
# Phase 3 Overrides
# ============================================================

# Will be set by CLI args
MODEL_NAME = None
CHECKPOINT_ROLLING_WINDOW = 15  # Keep last N checkpoints on disk


def phase3_model_factory(cfg):
    """Create model from pretrained weights. Works with any HF causal LM."""
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )


def tokenize_dataset(model_name, max_length=256, cache_dir='/tmp'):
    """Tokenize WikiText-103 with the target model's tokenizer. Cache to disk."""
    safe_name = model_name.replace('/', '_')
    cache_path = Path(cache_dir) / f'phase3_tokenized_{safe_name}.pt'

    if cache_path.exists():
        print(f'Loading cached tokenized data from {cache_path}')
        cached = torch.load(cache_path, weights_only=True)
        return cached['input_ids'], cached['attention_mask']

    print(f'Tokenizing WikiText-103 with {model_name} tokenizer...')
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')

    # Filter empty lines and tokenize
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    print(f'  {len(texts)} non-empty passages')

    # Tokenize in chunks
    all_ids = []
    all_masks = []
    chunk_size = 1000
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        encoded = tokenizer(
            chunk, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        all_ids.append(encoded['input_ids'])
        all_masks.append(encoded['attention_mask'])

        if (i // chunk_size) % 10 == 0:
            print(f'  Tokenized {min(i+chunk_size, len(texts))}/{len(texts)}...')

    input_ids = torch.cat(all_ids, dim=0)
    attention_mask = torch.cat(all_masks, dim=0)

    # Save cache
    torch.save({'input_ids': input_ids, 'attention_mask': attention_mask}, cache_path)
    print(f'  Cached {input_ids.shape[0]} samples to {cache_path}')

    return input_ids, attention_mask


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=CHECKPOINT_ROLLING_WINDOW):
    """Delete old checkpoints, keeping only the most recent N."""
    ckpt_files = sorted(ckpt_dir.glob('step_*.pt'))
    if len(ckpt_files) > keep_last_n:
        to_delete = ckpt_files[:-keep_last_n]
        for f in to_delete:
            f.unlink()


def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser(description='Phase 3: Scale Validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B',
                        help='HuggingFace model name')
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='/tmp')
    args = parser.parse_args()

    MODEL_NAME = args.model
    seed = args.seed
    safe_model = args.model.replace('/', '_')
    output_dir = Path(f'results/phase3/{safe_model}/seed{seed}')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    config = TrainingConfig(
        model_name=args.model,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=seed,
        device=device,
        use_amp=use_amp,
        skip_inline_leap=True,  # Save GPU memory — Pass 2 does full K-sweep from disk
    )

    print(f'Phase 3: Scale Validation')
    print(f'Model: {args.model}')
    print(f'Device: {device}')
    print(f'Seed: {seed}, Max steps: {config.max_steps}, Batch size: {config.batch_size}')
    print(f'Output: {output_dir}', flush=True)

    # Tokenize data with target model's tokenizer
    all_ids, all_mask = tokenize_dataset(args.model, max_length=args.max_length,
                                          cache_dir=args.cache_dir)

    train_ids = all_ids[:-VAL_HOLDOUT]
    train_mask = all_mask[:-VAL_HOLDOUT]
    val_ids = all_ids[-VAL_HOLDOUT:]
    val_mask = all_mask[-VAL_HOLDOUT:]

    # Smaller val batch for large models to avoid OOM on logits
    val_bs = min(VAL_BATCH_SIZE, 8)
    val_batch = {
        'input_ids': val_ids[:val_bs],
        'attention_mask': val_mask[:val_bs],
    }

    # Load model config
    print('Loading model config...')
    model_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    # For init_sd, we load the pretrained model once and extract state_dict
    print('Loading pretrained model for init weights...')
    init_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    init_sd = {k: v.clone() for k, v in init_model.state_dict().items()}
    del init_model
    gc.collect()
    print(f'Init weights: {len(init_sd)} tensors, '
          f'{sum(v.numel() for v in init_sd.values()) / 1e6:.0f}M params', flush=True)

    # Build probe set with target tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    canonical_probe_set = build_probe_set(tokenizer, num_probes=100, max_length=128, seed=0)

    # Monkey-patch model_factory in phase2c module so it uses our factory
    import phase2c_depth_sweep
    phase2c_depth_sweep.model_factory = phase3_model_factory

    # Pass 1: Training
    if not args.skip_training:
        run_pass1_training(config, seed, output_dir, model_cfg, init_sd,
                          train_ids, train_mask, val_batch, canonical_probe_set)

        # Rolling checkpoint cleanup
        ckpt_dir = output_dir / 'checkpoints'
        cleanup_old_checkpoints(ckpt_dir, keep_last_n=CHECKPOINT_ROLLING_WINDOW)
    else:
        print('Skipping Pass 1 (--skip-training)')

    del init_sd
    gc.collect()

    # Pass 2: K-sweep
    run_pass2_ksweep(config, seed, output_dir, model_cfg, val_batch)

    # Pass 3: Cascades
    run_pass3_cascades(config, seed, output_dir, model_cfg, train_ids, train_mask, val_batch)

    # Combine
    combined = combine_results(seed, output_dir)

    # Cleanup
    if args.cleanup:
        ckpt_dir = output_dir / 'checkpoints'
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f'Cleaned up {ckpt_dir}')

    # Plots
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
