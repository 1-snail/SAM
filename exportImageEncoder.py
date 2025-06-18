import torch
from segment_anything import sam_model_registry
import onnx

# 替换为你的模型类型和checkpoint路径
sam_checkpoint = "notebooks\sam_vit_h_4b8939.pth" # 假设是 ViT-H
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.eval() # 切换到评估模式

# 图像编码器的输入需要 1024x1024 的预处理图像
# 注意：这里需要传入和SamPredictor预处理一致的图像尺寸和归一化
# 通常SamPredictor内部会进行padding和resize到1024x1024
dummy_input_image = torch.randn(1, 3, 1024, 1024)

# 导出图像编码器
output_onnx_path_image_encoder = "sam_image_encoder.onnx"
torch.onnx.export(
    sam.image_encoder,
    dummy_input_image,
    output_onnx_path_image_encoder,
    input_names=["input_image"],   # <-- 图像编码器通常期望 "input_image"
    output_names=["image_embeddings"],
    opset_version=17, # 或者其他兼容的版本
    dynamic_axes={
        "input_image": {0: "batch_size"}, # 可选，如果需要动态batch_size
        "image_embeddings": {0: "batch_size"},
    },
)

print(f"Image encoder ONNX model saved to {output_onnx_path_image_encoder}")

# 再次检查你现有的 sam_onnx_example.onnx 的输入，确认它只接收 image_embeddings
# (你已经确认了，所以这一步可以跳过，或用于双重确认)
# onnx_model = onnx.load("D:\\study\\onnx\\UseOnnx\\SamOnnx\\sam_onnx_example.onnx")
# print("\nInput names of your existing sam_onnx_example.onnx:")
# for input_tensor in onnx_model.graph.input:
#     print(f"- {input_tensor.name}")