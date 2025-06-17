import onnx
import onnxruntime as ort

# 加载你的 ONNX 模型
onnx_model_path = "D:\\study\\onnx\\UseOnnx\\SamOnnx\\sam_onnx_example.onnx"
onnx_model = onnx.load(onnx_model_path)

print("ONNX Model Input Names:")
for input_tensor in onnx_model.graph.input:
    print(f"- {input_tensor.name}")

# 如果你想看更详细的，包括形状和数据类型
print("\nONNX Model Inputs (detailed):")
for input_tensor in onnx_model.graph.input:
    print(f"  Name: {input_tensor.name}")
    print(f"  Shape: {[dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in input_tensor.type.tensor_type.shape.dim]}")
    print(f"  Type: {onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_tensor.type.tensor_type.elem_type]}")

# 也可以用 onnxruntime 检查
sess = ort.InferenceSession(onnx_model_path)
print("\nONNX Runtime Session Input Names:")
for _input in sess.get_inputs():
    print(f"- {_input.name}")
    print(f"  Shape: {_input.shape}")
    print(f"  Type: {_input.type}")