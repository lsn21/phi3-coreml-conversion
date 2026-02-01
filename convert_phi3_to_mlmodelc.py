# convert_phi3_to_mlmodelc.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import coremltools as ct

# --- Настройки ---
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL_NAME = "Phi3Mini.mlmodelc"

print("🔄 Загружаем токенизатор и модель...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 🔥 ИСПРАВЛЕНИЕ: device_map=None — отключает accelerate
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map=None,  # ← ВАЖНО: отключает автоматическое распределение
    low_cpu_mem_usage=True,  # Рекомендуется для M1
)

# Создаем пример входа
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"✅ Входной тензор: {input_ids.shape}")

# Конвертируем модель в Core ML
print("🔄 Конвертируем модель в Core ML (это займет 5–10 минут)...")
mlmodel = ct.convert(
    model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=input_ids.shape,
            dtype=input_ids.dtype
        )
    ],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    skip_model_load=True
)

# Сохраняем модель
print(f"💾 Сохраняем модель как {OUTPUT_MODEL_NAME}...")
mlmodel.save(OUTPUT_MODEL_NAME)

print(f"🎉 Готово! Модель сохранена: {OUTPUT_MODEL_NAME}")
