import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

# Suppress warnings
torch._dynamo.config.suppress_errors = True

class JawirumInference:
    def __init__(self, model_path="jawirumi_v1_model"):
        # Get the model path relative to backend folder
        model_path = os.path.join(os.path.dirname(__file__), "../../", model_path)
        
        print("[LLM] Loading model...")
        self.model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("[LLM] Model loaded!")
    
    def transliterate(self, text: str, mode: str) -> str:
        """
        Translate Jawi ↔ Rumi using LLM
        
        Args:
            text: Input text
            mode: 'jawi2rumi' or 'rumi2jawi'
        
        Returns:
            Translated text
        """
        # Format prompt based on mode
        if mode == "jawi2rumi":
            prompt = f"<jawi2rumi> {text}"
        elif mode == "rumi2jawi":
            prompt = f"<rumi2jawi> {text}"
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids["input_ids"].to(self.model.device),
                attention_mask=input_ids["attention_mask"].to(self.model.device),
                max_new_tokens=128,
                do_sample=False,
            )
        
        # Decode and return
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

# Load once at startup
inferencer = JawirumInference()