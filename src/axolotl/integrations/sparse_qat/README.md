# Sparse QAT Training

Sparse QAT combines sparsity maintenance (keeping zero weights at zero) with Quantization-Aware Training (simulating low-precision arithmetic during training). This enables training models that are both sparse and quantized for maximum efficiency.

## Overview

This integration enables training models that are both:
- **Structurally sparse**: Maintaining pre-existing zero weight patterns from pruned models
- **Quantization-ready**: Trained with fake quantization for deployment efficiency

## Requirements

- Pre-sparsified base model (e.g., from [RedHatAI on Hugging Face](https://huggingface.co/RedHatAI) or a minimal 33K model [atomllama-33K-5x5-DigitMesh-sparse](https://huggingface.co/junzzhu/atomllama-33K-5x5-DigitMesh-sparse))
- `torchao >= 0.13.0`

---

## Quick Start

### Quantization-Aware Training (QAT)

Train a sparse model with quantization awareness for best accuracy, with [atomllama-33K-5x5-DigitMesh-sparse](https://huggingface.co/junzzhu/atomllama-33K-5x5-DigitMesh-sparse) as the example:

```bash
cd axolotl
accelerate launch -m axolotl.cli.train examples/sparse-qat/sparse_qat_atom.yaml
```


### Quantization

It may use either `axolotl` or `llm-compressor` (preferred to get safetensors output).

```bash
axolotl quantize examples/sparse-qat/sparse_qat_atom.yaml
```

or

```python
from transformers import AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
recipe = [
    QuantizationModifier(scheme="W4A8", targets="Linear", ignore=["lm_head"]),
]
model_name_or_path = "./models/atomllama-33K-5x5-DigitMesh-sparse-int8"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto", device_map="auto")
oneshot(
    model=model,
    recipe=recipe,
    output_dir="./models/atomllama-33K-5x5-DigitMesh-sparse-q8"
)
```

### Verification

Test the quantized model using the Hugging Face transformers library or vLLM. For sample testing code, see the [model card on Hugging Face of the quantized atomllama-33K-5x5-DigitMesh-sparse-q8](https://huggingface.co/junzzhu/atomllama-33K-5x5-DigitMesh-sparse-q8).

### Performance Comparison

Results on 5×5 digit mesh recognition (10 test patterns, digits 0-9):

```
============================================================
SUMMARY: Model Comparison
============================================================
Model                                              Accuracy        Avg Confidence
------------------------------------------------------------
atomllama-33K-5x5-DigitMesh                        10/10 (100.0%)      86.2%
atomllama-33K-5x5-DigitMesh-sparse                 10/10 (100.0%)      82.3%
atomllama-33K-5x5-DigitMesh-sparse-q8              10/10 (100.0%)      85.8%
============================================================
```

**Key Observations:**
- All three models maintain **100% accuracy** on the digit recognition task
- The sparse-q8 model achieves **85.8% average confidence**, recovering most of the confidence lost during sparsification (82.3%)
- The quantized sparse model provides **~3x compression** (46KB vs. 137KB) while maintaining accuracy with faster inference
