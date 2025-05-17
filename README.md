# Quantize TinyLlama CoreML on Mac ARM (May 2025)

**End-to-end workflow for quantizing a TinyLlama LLM CoreML model to int8 on Apple Silicon (M1/M2/M3). Script, logs, and all gotchas included.**

---

## 🚀 Context

Most open-source CoreML LLMs are only provided in float16. This means files are huge (2GB+), RAM usage is high, and iOS/macOS on-device deployment isn’t always efficient.
Official docs and forums rarely provide a working, up-to-date, step-by-step guide for post-conversion quantization (`.mlpackage`) on Mac ARM.

**This repo gives you a tested Python script, setup steps, and real logs from a successful int8 quantization (May 2025, Mac ARM, Python 3.11, coremltools 8.3.0).**

---

## 🛠️ Requirements

* **Mac ARM (Apple Silicon: M1, M2, M3)**
* **Python 3.11+** (install with `brew install python@3.11`)
* **coremltools >= 8.3.0** (`pip3 install --upgrade coremltools`)

---

## 📦 Quick Start

1. **Download your CoreML LLM**
   Example: [TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML](https://huggingface.co/TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML)

   ```bash
   huggingface-cli download --local-dir ./Models/TinyLlama TKDKid1000/TinyLlama-1.1B-Chat-v0.3-CoreML
   ```
2. **Copy `quantize_coreml.py` into the directory containing `float16_model.mlpackage`**
3. **Run the script:**

   ```bash
   python3.11 quantize_coreml.py
   ```

---

## 📝 Script

```python
import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

float16_model_path = "float16_model.mlpackage"
quant8_model_path  = "quant8_model.mlpackage"

print(f"🔄 Loading float16 model: {float16_model_path}")
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
```

---

## 📈 Results

* **File size reduction:** From \~2.2GB (`float16`) → much smaller (`quant8`)
* **Works on device:** iOS 17 / macOS Sonoma, Apple Silicon
* **int4 quantization:** Only available for iOS 18+ (will fail on current OS/devices)

---

## ⚠️ Notes and Troubleshooting

* Ignore warnings like “inf/-inf not supported by quantization. Skipped.”
* Quantization 4 bits (int4) is only supported on iOS 18+ and newer CoreML.
* If you get `ModuleNotFoundError: No module named 'coremltools.optimize.coreml'`, update coremltools and use Python 3.11+ on Mac ARM.
* Make sure you run the script **inside the directory containing** `float16_model.mlpackage` (or adapt the path).

---

## 🤝 Community

* Feel free to open issues or PRs!
* Share this repo on Reddit, Hugging Face forums, or Discord to help others.

---

**Created by [GreenBull31](https://github.com/GreenBull31) (Morgan CAMILLERI)**
