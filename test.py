import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load("./resources/models/unet/unet.onnx")
opset_version = model.opset_import[0].version
print(opset_version)

# convert model
model_simp, check = simplify(model)
