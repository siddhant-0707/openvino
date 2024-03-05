from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import openvino.torch

tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

compiled_model = torch.compile(model, backend="openvino")

with torch.no_grad():
    # The generate method can also be used directly if it works with the compiled model
    # Adjust generation parameters as needed
    # generated_ids = compiled_model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50)
    output = compiled_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
