import torch
import torchvision.models as models
import openvino.torch

# Assuming you have modified op_support.py as per the instructions to print debug statements for unsupported ops.

# Define or load your model
# For demonstration, we'll use a pre-defined model from torchvision
model = models.resnet18(pretrained=True)

# Prepare the model for evaluation (or set to .train() mode if evaluating training behaviour)
# model.eval()

# Sample input for the model; adjust size according to your model's requirement
# Here, a single image of 3 channels, 224x224 pixels is used as an example
input_tensor = torch.randn(1, 3, 224, 224)

# Compile the model with the OpenVINO backend
compiled_model = torch.compile(model, backend="openvino")

# Now you can run inference with the compiled model
# The debug statements for unsupported ops should print during the compilation step if there are any
output = compiled_model(input_tensor)

# Print the output or proceed with further processing
# print(output)
