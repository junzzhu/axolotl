# Sparse QAT Training

Sparse QAT combines sparsity maintenance (keeping zero weights at zero) with Quantization-Aware Training (simulating low-precision arithmetic during training). This enables training models that are both sparse and quantized for maximum efficiency.

## Overview

This integration enables training models that are both:
- **Structurally sparse**: Maintaining pre-existing zero weight patterns from pruned models
- **Quantization-ready**: Trained with fake quantization for deployment efficiency

## Requirements

- Pre-sparsified base model (e.g., from [RedHatAI on Hugging Face](https://huggingface.co/RedHatAI))
- `torchao >= 0.13.0`

---

## Quick Start

### Quantization-Aware Training (QAT)

Train a sparse model with quantization awareness for best accuracy:

```bash
cd axolotl
accelerate launch -m axolotl.cli.train examples/sparse-qat/sparse_qat_llama.yaml.yaml
```


### Quantization

It may use either `axolotl` or `llm-compressor`.

```bash
axolotl quantize examples/sparse-qat/sparse_qat_llama.yaml.yaml
```

or

```python
from transformers import AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
recipe = [
    QuantizationModifier(scheme="FP8_DYNAMIC", targets="Linear", ignore=["lm_head"]),
]
model_name_or_path = "./models/sparse-qat-llama3-8b"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
oneshot(
    model=model,
    recipe=recipe,
)
model.save_pretrained("./models/sparse-qat-llama3-8b-FP8", save_compressed=True, skip_sparsity_compression_stats=True)
```

### Deployment (with `RuntimeError`)

The quantized model from either of the above approaches has runtime errors still, i.e. `Engine core initialization` failure, which is to be fixed unfortunatly. Potentially, the QAT has bug still.
