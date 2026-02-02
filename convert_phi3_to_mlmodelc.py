import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import coremltools as ct
import numpy as np

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_MODEL_NAME = "Phi3Mini.mlpackage"  # ✅ ВАЖНО: .mlpackage для mlprogram

# 1. Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 2. Отключаем rope_scaling (обязательно!)
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hasattr(config, 'rope_scaling'):
    config.rope_scaling = None

# 3. Загружаем модель в fp32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map=None,
    low_cpu_mem_usage=True,
)

model.eval()
model = model.to("cpu")

# 4. Подготавливаем пример входа
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = inputs["input_ids"]      # [1, 128]
attention_mask = inputs["attention_mask"]  # [1, 128]

# 5. Обертка для трассировки
class Phi3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

wrapper = Phi3Wrapper(model)

# 6. Трассировка → TorchScript
traced_model = torch.jit.trace(
    wrapper,
    (input_ids, attention_mask),
    check_trace=False,
    strict=False
)

print("✅ TorchScript создан успешно!")

# 7. Конвертация в Core ML (mlprogram)
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int32),
    ],
    convert_to="mlprogram",  # ✅ Используем mlprogram
    compute_units=ct.ComputeUnit.ALL,
    skip_model_load=True,
)

# 8. Сохраняем с правильным расширением
mlmodel.save(OUTPUT_MODEL_NAME)  # ✅ .mlpackage — ОБЯЗАТЕЛЬНО!
print(f"🎉 Успешно сохранено: {OUTPUT_MODEL_NAME}")
