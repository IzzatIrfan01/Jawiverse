import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Suppress warnings
torch._dynamo.config.suppress_errors = True

print("GPU Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load model
print("\n[1] Loading model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "jawirumi_v1_model",
    device_map="auto",
    dtype=torch.float16,  # Changed from torch_dtype
)
tokenizer = AutoTokenizer.from_pretrained("jawirumi_v1_model")
print("✅ Model loaded!")

# Test inference
print("\n[2] Running inference...")
prompt = "چينتا سجاتي اداله سوات ڤركارا يڠ مرجوع كڤد حقيقة باطن"

# Simple tokenization
input_text = f"<jawi2rumi> {prompt}"
input_ids = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        input_ids["input_ids"].to(model.device),
        attention_mask=input_ids["attention_mask"].to(model.device),
        max_new_tokens=128,
        do_sample=False,  # Deterministic output
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n✅ Result:")
print(result)