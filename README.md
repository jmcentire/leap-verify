# Leap+Verify

**Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18739387.svg)](https://doi.org/10.5281/zenodo.18739387)

Leap+Verify applies speculative execution — predicting future model weights and validating predictions before acceptance — to accelerate neural network training. Inspired by [ASC (Automatically Scalable Computation)](https://dl.acm.org/doi/10.1145/2541940.2541985) and speculative decoding, the framework:

1. **Detects training regimes** (chaotic, transition, stable) using activation-space cosine similarity as a real-time Lyapunov proxy
2. **Predicts future weights** using analytic extrapolators (momentum, linear, quadratic)
3. **Validates before accepting** — rejected predictions have zero cost

## Key Findings

- **Universal momentum catastrophe**: Adam moment extrapolation fails at all scales (100–10,000× loss inflation)
- **Finite-difference predictors work**: Linear and quadratic weight extrapolation achieves 9–37% strict acceptance
- **Scale-dependent regime distribution**: Larger models spend more time in chaotic regimes (4% at 124M → 64% at 1.5B)
- **Larger models are more predictable when predictable**: 37% acceptance in Qwen 1.5B transition vs 9% in GPT-2 transition

## Repository Structure

```
src/
  speculative/
    predictors.py    # Momentum, linear, quadratic weight predictors
    workers.py       # Speculative worker simulation and cascades
    landscape.py     # Loss landscape probing
  fingerprint/
    capture.py       # Activation fingerprint computation
  regime/
    detect.py        # Three-regime classification
paper/
  leap_verify.tex    # Paper source
  references.bib     # Bibliography
phase2c_depth_sweep.py    # GPT-2 124M experiments
phase3_scale_validation.py # Qwen 2.5-1.5B experiments
```

## Reproducing Experiments

### Requirements

```bash
pip install torch transformers datasets
```

### GPT-2 124M (Phase 2c)

```bash
# Prepare data
python -c "
from transformers import GPT2LMHeadModel, AutoConfig, AutoTokenizer
from datasets import load_dataset
import torch, os
os.makedirs('/tmp/gpt2_local', exist_ok=True)
model = GPT2LMHeadModel.from_pretrained('gpt2')
config = AutoConfig.from_pretrained('gpt2')
config.save_pretrained('/tmp/gpt2_local')
torch.save(model.state_dict(), '/tmp/gpt2_local/model.pt')
tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
texts = [t for t in ds['text'] if len(t.strip()) > 50]
enc = tok(texts, max_length=256, truncation=True, padding='max_length', return_tensors='pt')
torch.save({'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}, '/tmp/phase0_tokenized.pt')
"

# Run experiment (GPU recommended)
python phase2c_depth_sweep.py --seed 42 --max-steps 2000 --device cuda --use-amp
```

### Qwen 2.5-1.5B (Phase 3)

```bash
python phase3_scale_validation.py --seed 42 --max-steps 2000 --device cuda --use-amp
```

## Citation

```bibtex
@software{mcentire2026leapverify,
  author = {McEntire, Jeremy},
  title = {Leap+Verify: Regime-Adaptive Speculative Weight Prediction for Accelerating Neural Network Training},
  year = {2026},
  doi = {10.5281/zenodo.18739387},
  url = {https://github.com/jmcentire/leap-verify}
}
```

## Acknowledgments

- **Amos Waterland** — Creator of the ASC architecture; formative discussions on trajectory-based speculative computation
- **Brian Cremeans** — Discussions on eigenvalue sign toggling in Lorenz system dynamics
- **Claude (Anthropic)** — Experimental implementation, GPU infrastructure, data analysis, and manuscript preparation

## License

MIT
