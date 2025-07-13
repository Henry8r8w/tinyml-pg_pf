import torch
import torch.quantization
import onnx
from perceptron import OneLMLP

# Load trained model
model = OneLMLP()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Quantization config
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # For CPU, adjust for ARM
model_prepared = torch.quantization.prepare(model, inplace=True)


with torch.no_grad():
    for i in range(0, 50, 32):  # Batch size 32
        x = torch.tensor(X_train[i:i+32], dtype=torch.float32)
        model_prepared(x)

torch.quantization.convert(model, inplace=True)

def quant_onnx(model):
    # Export to ONNX
    dummy_input = torch.randn(1, 13)
    torch.onnx.export(model, dummy_input, 'mlp_heart_quantized.onnx', 
                                input_names=['input'], output_names=['output'])
    print(model.layer_1.weight().dtype) # expects int8
    print(model.layer_1.weight().dtype) 

quant_onnx(model)