# convert_phi3_to_mlmodelc.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import coremltools as ct

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL_NAME = "Phi3Mini.mlmodelc"

print("🔄 Загружаем токенизатор и конфиг...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 🔧 КЛЮЧЕВОЙ ФИКС: УБИРАЕМ rope_scaling — он не нужен и вызывает ошибку!
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Убираем rope_scaling полностью — это ВАЖНО!
if hasattr(config, 'rope_scaling'):
    config.rope_scaling = None  # ← ФИКС: УБИРАЕМ ВСЁ, что вызывает ошибку

print("✅ Конфиг обновлён: rope_scaling = None")

# Загружаем модель с исправленным конфигом
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
)

# Пример входа
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"✅ Входной тензор: {input_ids.shape}")

# Конвертация в Core ML
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

# Сохраняем
print(f"💾 Сохраняем модель как {OUTPUT_MODEL_NAME}...")
mlmodel.save(OUTPUT_MODEL_NAME)

print(f"🎉 Готово! Модель сохранена: {OUTPUT_MODEL_NAME}")
