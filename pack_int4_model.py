import tensorflow as tf
import numpy as np
import os
import sys

try:
    from tensorflow.lite.python.schema_py_generated import Model
    from tensorflow.lite.python.schema_py_generated import BuiltinOperator
    from tensorflow.lite.python.schema_py_generated import TensorType

    print("Successfully imported TFLite FlatBuffer schema")
except ImportError:
    print("Failed to import TFLite FlatBuffer schema")
    sys.exit(1)

MODEL_ALPHA = 0.25
INPUT_SHAPE = 96
MODEL_DIR = f"models_mobilenet_v1_{INPUT_SHAPE}"
TFLITE_INT8_MODEL_PATH = os.path.join(
    MODEL_DIR, f"mobilenet_v1_{INPUT_SHAPE}_int8.tflite"
)
TFLITE_INT4_PACKED_PATH = os.path.join(
    MODEL_DIR, f"mobilenet_v1_{INPUT_SHAPE}_int4_packed.tflite"
)


def quantize_and_pack_weights(tensor, buffer_obj):
    """
    int8 -> fp32 -> int4(symmetric) -> packed int4
    """
    # 获取底层 bytes (numpy 视图)
    try:
        original_buffer_view = buffer_obj.DataAsNumpy().view(np.uint8)
    except AttributeError:
        print(" > Buffer does not support DataAsNumpy(), skip.")
        return

    int8_weights = original_buffer_view.view(np.int8)
    num_weights = len(int8_weights)
    if num_weights == 0:
        print(" > Empty weights, skip.")
        return

    q_params = tensor.Quantization()
    if (
        q_params is None
        or q_params.ScaleLength() == 0
        or q_params.ZeroPointLength() == 0
    ):
        print(f" > No quant params for {tensor.Name()}, skip.")
        return

    int8_scale = q_params.Scale(0)
    int8_zero_point = q_params.ZeroPoint(0)
    if int8_scale == 0:
        print(" > Scale=0, skip.")
        return

    fp32_weights = (int8_weights.astype(np.float32) - int8_zero_point) * int8_scale

    max_abs_val = np.max(np.abs(fp32_weights))
    if max_abs_val == 0:
        print(" > All-zero weights, skip.")
        return

    int4_scale = max_abs_val / 7.0
    if int4_scale == 0:
        print(" > int4 scale=0, skip.")
        return

    int4_weights = np.round(fp32_weights / int4_scale)
    int4_weights = np.clip(int4_weights, -8, 7).astype(np.int8)

    packed_len = (num_weights + 1) // 2
    packed_weights = np.zeros(packed_len, dtype=np.uint8)

    for i in range(packed_len):
        low_nibble = int4_weights[i * 2] & 0x0F
        high_nibble = 0
        if (i * 2 + 1) < num_weights:
            high_nibble = int4_weights[i * 2 + 1] & 0x0F
        packed_weights[i] = (high_nibble << 4) | low_nibble

    # 先清零再写入
    original_buffer_view[:] = 0
    original_buffer_view[0:packed_len] = packed_weights

    print(f" > Success: {num_weights} int8 -> {packed_len} packed int4 bytes.")


def post_process_to_int4(input_path, output_path):
    print("--- 开始 INT4 打包后处理 ---")
    print(f"加载 INT8 基线模型: {input_path}")

    with open(input_path, 'rb') as f:
        mutable_buffer = bytearray(f.read())

    model = Model.GetRootAsModel(mutable_buffer, 0)

    # 查找算子 opcode
    conv_op_index = -1
    dw_conv_op_index = -1
    for i in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(i)
        if op_code is None:
            continue

        builtin_code = op_code.BuiltinCode()
        if builtin_code == BuiltinOperator.CONV_2D:
            conv_op_index = i
        elif builtin_code == BuiltinOperator.DEPTHWISE_CONV_2D:
            dw_conv_op_index = i

    if conv_op_index == -1 or dw_conv_op_index == -1:
        print("错误：缺少 CONV_2D 或 DEPTHWISE_CONV_2D。")
        return

    print(f"识别码: CONV_2D=[{conv_op_index}], DEPTHWISE_CONV_2D=[{dw_conv_op_index}]")

    # get the first subgraph
    if model.SubgraphsLength() == 0:
        print("Error: model doesn't include subgraphs")
        return
    subgraph = model.Subgraphs(0)
    if subgraph is None:
        print("Error: cannot access subgraph[0]")
        return

    # find all weights to be modified
    tensors_to_pack = set()  # use set to eliminate repeated ones

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        if op is None:
            continue

        op_code_index = op.OpcodeIndex()

        # check if we want this operator
        if op_code_index == conv_op_index or op_code_index == dw_conv_op_index:
            # normally weights index=1
            if op.InputsLength() > 1:
                weight_tensor_index = op.Inputs(1)
                tensors_to_pack.add(weight_tensor_index)
            else:
                continue

    print(
        f"在 {subgraph.OperatorsLength()} 个算子中，识别出 {len(tensors_to_pack)} 个唯一的权重张量需要打包。"
    )

    for tensor_index in tensors_to_pack:
        tensor = subgraph.Tensors(tensor_index)
        if tensor is None:
            print(f"Warning: can not access tensor {tensor_index}, skip.")
            continue

        tensor_name = tensor.Name()
        if tensor_name is None:
            print(f"Warning: can not access tensor Name for {tensor_index}, skip")
            continue

        name = (
            tensor_name.decode('utf-8') if tensor.Name() else f"tensor_{tensor_index}"
        )

        # weights tensor restored in buffer list
        buffer_index = tensor.Buffer()
        buffer_obj = model.Buffers(buffer_index)
        if buffer_obj is None or buffer_obj.DataLength() == 0:
            print(f"警告：tensor {tensor_name} 的缓冲区 {buffer_index} 为空。跳过。")
            continue

        print(f"处理中: {name} (Buffer {buffer_index})")
        quantize_and_pack_weights(tensor, buffer_obj)

    with open(output_path, 'wb') as f:
        f.write(mutable_buffer)

    print("--- INT4 打包完成 ---")
    print(f"已保存: {output_path}")


# --- 主函数 ---
if __name__ == "__main__":

    # 1. 检查 INT8 基线模型是否存在
    if not os.path.exists(TFLITE_INT8_MODEL_PATH):
        print(f"错误: {TFLITE_INT8_MODEL_PATH} 不存在。")
        print("请先运行你的第一个脚本来生成 INT8 基线模型。")
        sys.exit(1)

    # 2. 执行后处理
    post_process_to_int4(TFLITE_INT8_MODEL_PATH, TFLITE_INT4_PACKED_PATH)

    # 3. 比较文件大小
    int8_size = os.path.getsize(TFLITE_INT8_MODEL_PATH) / (1024)
    int4_size = os.path.getsize(TFLITE_INT4_PACKED_PATH) / (1024)

    print("\n--- 文件大小比较 ---")
    print(f"原始 INT8 模型: {int8_size:.1f} KB")
    print(f"打包 INT4 模型: {int4_size:.1f} KB")
    print(f"注意：文件大小应该完全一样 ({int4_size:.1f} KB)，")
    print(f"因为我们只是用打包数据（和 0）覆盖了原始缓冲区。")
