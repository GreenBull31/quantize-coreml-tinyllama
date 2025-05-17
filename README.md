# Quantize TinyLlama CoreML on Mac ARM (May 2025)

**End-to-end workflow to quantize a TinyLlama LLM CoreML model to int8 on Apple Silicon.**

---

## 🚦 Context

Most CoreML LLM models are only available in float16, which results in large file sizes (2+ GB) and high RAM use. Community and Apple docs are still vague on post-conversion quantization for `.mlpackage` on Mac ARM, and many devs believe it’s “not possible”.

**This repo provides a working Python script, setup, and real logs from a successful int8 quantization (May 2025, Mac ARM, Python 3.11, coremltools 8.3.0).**

---

## 📦 Requirements

- **Mac ARM (Apple Silicon, M1/M2/M3)**
- **Python 3.11+** (`brew install python@3.11`)
- **coremltools 8.3.0+** (`pip3 install --upgrade coremltools`)

---

## 🔥 Quick Start

1. **Download a CoreML LLM** (ex: [TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML](https://huggingface.co/TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML)):
    ```bash
    huggingface-cli download --local-dir ./Models/TinyLlama TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML
    ```
2. **Copy `quantize_coreml.py` in the directory containing your `float16_model.mlpackage`**
3. **Run the script**:
    ```bash
    python3.11 quantize_coreml.py
    ```

---

## 🛠️ Script

```python
import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

float16_model_path = "float16_model.mlpackage"
quant8_model_path  = "quant8_model.mlpackage"

print(f"🔄 Loading float16 model : {float16_model_path}")
model = ct.models.MLModel(float16_model_path)

op_config = OpLinearQuantizerConfig(mode="linear_symmetric")
config = OptimizationConfig(global_config=op_config)

print("⚡ Quantizing to 8 bits (int8)...")
try:
    quant8_model = linear_quantize_weights(model, config=config)
    quant8_model.save(quant8_model_path)
    print(f"✅ Quantized int8 model saved: {quant8_model_path}")
except Exception as e:
    print(f"❌ Quantization failed: {e}")
