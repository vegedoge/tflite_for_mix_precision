import tensorflow as tf
import keras
import numpy as np
import os


# try mobileNet V2 first
model = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), alpha=1.0, include_top=False, weights='imagenet'
)

# summarize the model
model.summary()


# representative dataset generator
def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 96, 96, 3)
        yield [data.astype(np.float32)]


# converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# opt flags, default includes int8 quantization
converter.optimizations = {tf.lite.Optimize.DEFAULT}

# set representative dataset for int8 quantization
converter.representative_dataset = representative_data_gen  # type: ignore

# enforce full int8 quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

tflite_model = converter.convert()

# 输出到与模型匹配的目录
out_dir = "models_mobilenet_v2_96"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "mobilenet_v2_96_int8.tflite")

with open(out_path, 'wb') as f:
    f.write(tflite_model)  # type: ignore

print(f"tflite model saved: {out_path}")
