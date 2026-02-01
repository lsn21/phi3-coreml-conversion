# convert_phi3_to_mlmodelc.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import coremltools as ct
import numpy as np

# Проверка версии NumPy — критично для coremltools
print(f"✅ NumPy version: {np.__version__}")
if int(np.__version__.split('.')[0]) >= 2:
    raise RuntimeError("❌ NumPy 2.0+ не поддерживается coremltools 7.0. Используйте numpy==1.26.4")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL_NAME = "Phi3Mini.mlmodelc"

print("🔄 Загружаем токенизатор и конфиг...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, 'rope_scaling'):
    config.rope_scaling = None  # 🔧 ФИКС: Убираем rope_scaling
print("✅ Конфиг обновлён: rope_scaling = None")

# 🔧 ОБЯЗАТЕЛЬНО: low_cpu_mem_usage=True требует accelerate
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
)

# ✅ ПРИМЕР ВХОДА — КЛЮЧЕВОЙ ФИКС: int64 → int32
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(torch.int32)  # ✅ ВАЖНО: преобразуем в int32!
print(f"✅ Входной тензор: {input_ids.shape}, dtype={input_ids.dtype}")

# Конвертация в Core ML
print("🔄 Конвертируем модель в Core ML (это займет 5–10 минут)...")
mlmodel = ct.convert(
    model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=input_ids.shape,
            dtype=input_ids.dtype  # ✅ Теперь это torch.int32 — поддерживается!
        )
    ],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    skip_model_load=True
)

# Сохраняем
print(f"💾 Сохраняем модель как {OUTPUT_MODEL_NAME}...")
mlmodel.save(OUTPUT_MODEL_NAME)

print(f"🎉 Готово! Модель сохранена: {OUTPUT_MODEL_NAME}")
