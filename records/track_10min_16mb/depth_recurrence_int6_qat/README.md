# Depth Recurrence + Int6 QAT

## Summary

This submission explores **depth recurrence** as a parameter-efficient alternative to unique transformer layers. Instead of N unique blocks, we use K unique blocks looped L times each (K*L = effective depth), with per-loop learnable "level signals" to differentiate each pass (inspired by RingFormer). Combined with **int6 quantization-aware training (QAT)**, this allows packing significantly more effective capacity into the 16MB artifact budget.

## Key Ideas

### 1. Depth Recurrence (3 blocks x 4 loops = 12 effective layers)
- Only 3 unique transformer blocks are stored in the artifact
- Each block is reused 4 times in sequence, giving 12 effective layers of depth
- Per-loop level signals (small learnable vectors added before each block invocation) allow the shared blocks to behave differently at each depth position
- U-Net skip connections are maintained across the full effective depth

### 2. Int6 Quantization-Aware Training
- During training, all linear layer weights pass through a fake-quantize function simulating int6 precision ([-31, 31] range) via straight-through estimator
- The model learns to be robust to 6-bit precision from the start
- At export, weights are quantized to the same int6 range (stored in int8 containers) and zlib-compressed
- Combined with depth recurrence, this gives extreme compression: few unique parameters, each stored in 6 bits

### 3. Wider Architecture
- With only 3 unique blocks to store, parameter budget is concentrated into wider layers
- model_dim=2048, 32 attention heads, 3x MLP expansion
- ~106M total parameters, but only ~35M unique stored parameters

### 4. Additional Techniques
- **Muon optimizer with weight decay (0.04)**: Better regularization
- **Sliding window evaluation (stride=64)**: Each scored token gets full context window
- **Per-loop level signals**: Low-rank differentiation for weight-shared layers

## Architecture Details

| Component | Value |
|-----------|-------|
| Unique blocks | 3 |
| Loops per block | 4 |
| Effective depth | 12 layers |
| Model dimension | 2048 |
| Attention heads | 32 |
| KV heads | 4 (GQA) |
| MLP multiplier | 3x |
| Head dimension | 64 |
| Vocabulary | 1024 (SentencePiece BPE) |
| Tied embeddings | Yes |
| Quantization | Int6 QAT (STE) |
| Compression | zlib level 9 |

## Results

| Run | val_bpb | Compressed Size | Training Time |
|-----|---------|----------------|---------------|
| Run 1 | TBD | TBD | TBD |
| Run 2 | TBD | TBD | TBD |
| Run 3 | TBD | TBD | TBD |

## Why Depth Recurrence?

The Parameter Golf challenge constrains artifact size (16MB) but not compute. Depth recurrence trades compute (more forward passes through shared blocks) for parameters (fewer unique weights to store). This is the opposite trade-off from standard architectures that maximize unique parameters.

At the extreme, a single transformer block looped N times uses 1/N the parameters of N unique blocks while performing the same number of FLOPs. The level signals add negligible parameter cost (<0.1%) while allowing each loop iteration to specialize.

## References

- [RingFormer: A Single Transformer Block is Sufficient](https://arxiv.org/abs/2502.13181)
- [SpiralFormer: Multi-Resolution Recursive Transformers](https://arxiv.org/abs/2602.11698)
- [Mixture-of-Recursions: Adaptive Depth](https://arxiv.org/abs/2507.10524)
