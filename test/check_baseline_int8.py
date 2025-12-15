import tensorflow as tf
import numpy as np
import os
from PIL import Image

# 1. 设置路径
MODEL_PATH = "../models_mobilenet_v1_96/mobilenet_v1_96_int8.tflite"
# 确保这里用的图片和你 MCU 代码里硬编码的图片数据是一模一样的！
# 如果 MCU 用的是全 0 或随机数，这里也要用全 0 或随机数。
# 假设 MCU 用的是一张真实的 airplane 图片数据：
IMAGE_PATH = "test_image_airplane.jpg"  # 或者用 np.random 模拟


def load_and_preprocess_image(image_path):
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    img = img.resize((96, 96))  # 调整到模型输入大小

    # 转换为 numpy 数组
    img_array = np.array(img)

    # 归一化到 [-1, 1]
    img_array = (img_array.astype(np.float32) - 127.5) / 127.5

    # 添加 batch 维度
    img_array = np.expand_dims(img_array, axis=0)

    return img_array.astype(np.int8)  # 模型是 INT8


def run_inference():
    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 准备输入数据
    input_shape = input_details[0]['shape']

    # --- 这里要非常注意 ---
    # 必须确保输入数据与 MCU 完全一致
    # 这里为了演示，我生成一个全 1 的输入，或者你可以手动填入 MCU 里的数据
    input_data = np.ones(input_shape, dtype=np.int8)

    # 更好的做法：如果你 MCU 里有一张图，把它的像素值复制过来

    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 推理
    interpreter.invoke()

    # 获取输出 (Raw Logits, INT8 类型)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print("\n--- Python INT8 Baseline output (Raw Logits) ---")
    print(f"Output Shape: {output_data.shape}")
    print("Values (first 10 values):")
    print(output_data[0, :10])

    # 如果你也想看 Softmax 后的概率
    scale, zero_point = output_details[0]['quantization']
    dequantized = (output_data.astype(np.float32) - zero_point) * scale
    import scipy.special

    probs = scipy.special.softmax(dequantized)
    print("\nProbabilities:")
    print(probs[0, :10])


if __name__ == "__main__":
    run_inference()
