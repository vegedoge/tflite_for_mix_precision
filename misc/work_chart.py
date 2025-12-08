import graphviz
from IPython.display import display

# 创建一个新的有向图
dot = graphviz.Digraph(
    'TFLM_INT4_Optimization_Flow', comment='TFLM Int4 Packing and Optimization Flow'
)

# 设置全局图表属性，使图表更专业
dot.attr(
    rankdir='LR',
    size='12,8',
    ratio='fill',
    splines='ortho',
    pad='0.5',
    bgcolor='#F9F9F9',
)

# 设置节点和边的默认属性
dot.attr(
    'node',
    shape='box',
    style='filled,rounded',
    fontname='Helvetica',
    fontsize='12',
    margin='0.2',
)
dot.attr('edge', fontname='Helvetica', fontsize='10', color='#555555', arrowhead='open')

# --- 定义节点 ---

# 1. 输入和中间产物 (Artifacts - 灰色文档形状)
dot.attr('node', shape='note', style='filled', fillcolor='#E8E8E8', color='#999999')
dot.node('A1', 'Original TFLite Model\n(int8 quantized)')
dot.node('A2', 'Packed TFLite Model\n(int8 -> int4 weights)')
dot.node('A3', 'Modified TFLM Library\n(Source Code)')

# 2. Host PC 端离线处理步骤 (Host Side Processes - 蓝色)
dot.attr(
    'node', shape='box', style='filled,rounded', fillcolor='#DAE8FC', color='#6C8EBF'
)
dot.node('P1', 'Step 1: Model Packing Script\n(Pack int8 weights into int4 container)')
dot.node(
    'P3',
    'Step 3: Modify TFLM Library\n(Adapt kernel logic to unpack/interpret int4 data)',
)

# 3. Target Board 端嵌入式步骤 (Target Side Processes - 橙色)
dot.attr(
    'node', shape='box', style='filled,rounded', fillcolor='#FFE6CC', color='#D79B00'
)
dot.node('P2', 'Step 2: Customize Ops on Board\n(Low-level Optimized Kernels for HW)')
dot.node(
    'P4',
    'Step 4: Integrate onto Board Firmware\n(Link modified TFLM & Custom Ops into project)',
)

# 4. 决策和测试 (Testing & Decision - 紫色/绿色)
dot.attr('node', shape='diamond', style='filled', fillcolor='#D5E8D4', color='#82B366')
dot.node('D1', 'Success Confirmation\n(Does the packing scheme work?)')

dot.attr(
    'node',
    shape='box',
    style='filled,rounded,bold',
    fillcolor='#E1D5E7',
    color='#9673A6',
)
dot.node(
    'P5',
    'Step 5: Testing & Validation\n(Benchmark on various models.\nNote: Limited Ops supported currently)',
)

# --- 定义图结构 (使用子图 Cluster 来分组) ---

# Cluster 1: Host Side Activities
c = graphviz.Digraph(name='cluster_host')
c.attr(
    style='dashed',
    color='#6C8EBF',
    label='Host PC (Offline Development & Prep)',
    fontcolor='#6C8EBF',
    bgcolor='#F4F7FC',
)
c.edges([('A1', 'P1'), ('P1', 'A2'), ('A2', 'D1'), ('D1', 'P3'), ('P3', 'A3')])
# 如果失败的反馈循环
c.edge('D1', 'P1', label='No, rethink scheme', style='dotted', color='red')
dot.subgraph(c)

# Cluster 2: Target Board Activities
c = graphviz.Digraph(name='cluster_target')
c.attr(
    style='dashed',
    color='#D79B00',
    label='Target Board (Embedded Implementation)',
    fontcolor='#D79B00',
    bgcolor='#FFF8F0',
)
c.node('P2')
c.node('P4')
c.node('P5')
# 定义内部连接
c.edge('P2', 'P4', label='Provide optimized kernels')
c.edge('P4', 'P5', label='Flash Firmware')
dot.subgraph(c)


# --- 定义跨区域连接 (Critical Links) ---

# P3 (Modify Lib) 需要知道 P2 (Custom Ops) 的接口定义
dot.edge(
    'P2',
    'P3',
    label='Define Operator Interface/API definition',
    style='dashed',
    constraint='false',
)

# A3 (Modified Lib Source) 需要被集成到 P4 (Board Firmware)
dot.edge('A3', 'P4', label='Cross-compile & Link')

# A2 (Packed Model) 是 P5 (Testing) 的输入数据
dot.edge('A2', 'P5', label='Load Model Data to Board', weight='2')

# P5 测试结果反馈到开发流程
dot.edge(
    'P5', 'P2', label='Performance feedback\n(Iterate optimization)', style='dotted'
)
dot.edge('P5', 'P3', label='Bug report / Feature request', style='dotted')


# 渲染并显示图像
display(dot)

# save pdf
dot.render('tflm_int4_optimization_flow', format='pdf', cleanup=True)
