import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Suppress Triton/Inductor warnings on laptop GPU
torch._dynamo.config.suppress_errors = True

print("GPU Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load model
print("\n[1] Loading model...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "jawirumi_v1_model",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("jawirumi_v1_model")
print("✅ Model loaded!")

# Test inference
print("\n[2] Running inference...")
prompt = "چينتا سجاتي اداله سوات ڤركارا يڠ مرجوع كڤد حقيقة باطن"

messages = [{"role": "user", "content": f"<jawi2rumi> {prompt}"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        temperature=0.1,
        do_sample=False,
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n✅ Result:")
print(result)