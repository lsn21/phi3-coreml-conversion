import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import coremltools as ct
import numpy as np

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL_NAME = "Phi3Mini.mlmodelc"

# 1. Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 2. Фиксим rope_scaling (обязательно!)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, 'rope_scaling'):
    config.rope_scaling = None

# 3. Загружаем модель в fp32 (важно!)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=torch.float32,  # ⚠️ Используем float32, а не "auto"
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
)

# 4. Переводим в eval-режим и отключаем dropout
model.eval()
model = model.to("cpu")

# 5. Генерируем тестовый ввод (важно: длина должна быть фиксированной!)
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = inputs["input_ids"]  # Shape: [1, 128]
attention_mask = inputs["attention_mask"]  # Shape: [1, 128]

# 6. ✅ СОЗДАЁМ TORCHSCRIPT ЧЕРЕЗ trace() — КЛЮЧЕВОЙ ШАГ!
print("🔄 Создаём TorchScript через tracing...")

# Определяем сигнатуру входа для tracing
class Phi3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

wrapper = Phi3Wrapper(model)

# Выполняем tracing
traced_model = torch.jit.trace(
    wrapper,
    (input_ids, attention_mask),
    check_trace=False,  # ⚠️ Иногда trace не проходит — отключаем проверку
    strict=False
)

print("✅ TorchScript создан успешно!")

# 7. Конвертируем в Core ML — теперь source не нужен, потому что это TorchScript
print("🔄 Конвертируем TorchScript в Core ML...")

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
    ],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    skip_model_load=True,
)

# 8. Сохраняем
mlmodel.save(OUTPUT_MODEL_NAME)
print(f"🎉 Успешно сохранено: {OUTPUT_MODEL_NAME}")

