import coremltools as ct
from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
import os

# --- Paramètres ---
base_dir = "Models/TinyLlama/coreml/text-generation"
# Nouveau chemin SANS Models/TinyLlama/coreml/text-generation devant !
float16_model_path = "float16_model.mlpackage"
quant8_model_path  = "quant8_model.mlpackage"
quant4_model_path  = os.path.join(base_dir, "quant4_model.mlpackage")

# --- Chargement du modèle CoreML existant (float16) ---
print(f"🔄 Chargement du modèle float16 : {float16_model_path}")
model = ct.models.MLModel(float16_model_path)

# --- Configuration de la quantification ---
op_config = OpLinearQuantizerConfig(mode="linear_symmetric")
config = OptimizationConfig(global_config=op_config)

# --- Quantification 8 bits (INT8) ---
print("⚡ Quantification 8 bits (int8)...")
try:
    quant8_model = linear_quantize_weights(model, config=config)
    quant8_model.save(quant8_model_path)
    print(f"✅ Modèle quantifié 8 bits sauvegardé : {quant8_model_path}")
except Exception as e:
    print(f"❌ Quantification 8 bits échouée : {e}")

# --- Quantification 4 bits (INT4) ---
print("⚡ Quantification 4 bits (int4)...")
try:
    op_config_4bit = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)
    config_4bit = OptimizationConfig(global_config=op_config_4bit)
    quant4_model = linear_quantize_weights(model, config=config_4bit)
    quant4_model.save(quant4_model_path)
    print(f"✅ Modèle quantifié 4 bits sauvegardé : {quant4_model_path}")
except Exception as e:
    print(f"❌ Quantification 4 bits échouée : {e}")

print("🎉 Script terminé. Vérifie la taille des fichiers générés et teste-les dans Xcode !")