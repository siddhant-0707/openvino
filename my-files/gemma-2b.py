from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openvino.torch

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

# Prepare the model for inference
model.eval()

# Example text
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Compile the model with the OpenVINO backend
# Note: Adjust 'example_inputs' to match your actual input structure
compiled_model = torch.compile(model, backend="openvino")

# Generate text with the compiled model
with torch.no_grad():
    # The generate method can also be used directly if it works with the compiled model
    # Adjust generation parameters as needed
    # generated_ids = compiled_model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50)
    output = compiled_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

# Decode generated ids to text
# generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print(generated_text)
