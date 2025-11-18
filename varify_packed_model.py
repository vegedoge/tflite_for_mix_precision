import tensorflow as tf
import numpy as np
import os

# 设置你的 packed 模型路径
MODEL_PATH = "models_mobilenet_v1_96/mobilenet_v1_96_int4_packed.tflite"

try:
    # 尝试加载模型
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    print("模型加载成功。开始检查权重...")

    # 遍历所有 tensor
    details = interpreter.get_tensor_details()
    found_packed = False

    for tensor in details:
        # 简单的启发式检查：名字里带 'Conv2D' 且维度大于 1 的通常是权重
        if 'Conv2D' in tensor['name'] and len(tensor['shape']) > 1:
            # 获取权重数据
            # 注意：TFLite Python API 会阻止你直接读取"常量" tensor
            # 我们通常需要通过 tensor index 强行读取
            try:
                data = interpreter.tensor(tensor['index'])()
            except:
                continue  # 或者是只读的 buffer，这里简单跳过，实际需要解析 Flatbuffer

            # 在 Python API 中很难直接拿到原始 buffer，
            # 最直观的方法其实是直接看文件二进制。
            pass

    # --- 更直接的方法：直接读取二进制文件并寻找特征 ---
    with open(MODEL_PATH, "rb") as f:
        content = f.read()

    # 这是一个非常粗略的检查：
    # 如果我们成功打包了，文件里应该会出现大段大段的连续 0x00。
    # MobileNet 的权重通常很大，所以会有几 KB 的 0。

    zeros_block = b'\x00' * 1024  # 寻找连续 1024 个 0
    if zeros_block in content:
        print("验证通过：在模型文件中发现了大块的连续零区域。")
        print("这极大概率意味着 Pack 操作成功，且保留了 Padding。")
    else:
        print("警告：未发现大块连续零。可能 Pack 失败或模型太小。")

except Exception as e:
    print(f"验证出错: {e}")
