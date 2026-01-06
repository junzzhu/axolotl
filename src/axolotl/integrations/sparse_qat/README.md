# Unified Approach for Sparse + QAT Training

This integration unifies Sparse Fine-Tuning and Quantization-Aware Training (QAT) into a **single, efficient training run**, with the intention to streamline the fine-tuning and compression of deployable LLMs.

## Overview

This integration enables training models that are both:
- **Structurally sparse**: Maintaining pre-existing zero weight patterns from pruned models
- **Quantization-ready**: Trained with fake quantization for deployment efficiency

Combining 50% sparsity with INT4 quantization achieves:
- **~8x memory reduction** (50% weights × 4 bits vs 16 bits)
- **2.5-3× inference speedup** (skip zeros + reduced precision ops)
- **Single training run** (vs 2-3 sequential runs)

## Requirements

- Pre-sparsified base model (e.g., from [RedHatAI on Hugging Face](https://huggingface.co/RedHatAI))
- `torchao >= 0.13.0`

---

## Architecture Design: Hybrid TorchAO + Sparsity Approach

This implementation uses a **hybrid architecture** that separates concerns for robust combined sparse and QAT training:

**TorchAO's Native QAT** (handles quantization):
- Exploits `FakeQuantizedLinear` and `FakeQuantizedEmbedding` layers.
- Utilizes TorchAO's optimized CUDA kernels for efficient fake quantization.
- Benefits from automatic updates and improvements within the TorchAO library.

**FSDP-Safe Sparsity Enforcement** (handles sparsity):
- Extracts static sparsity masks from pre-pruned models during initialization.
- Applies masks after each optimizer step.
- Independent of quantization strategy.

---

## Quick Start

TBD