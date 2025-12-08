import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------
# 1. 数据准备 (Data Preparation)
# ---------------------------------------------------------
log_data = """
CONV_2D took 27797 ticks (27 ms)
DEPTHWISE_CONV_2D took 10470 ticks (10 ms)
CONV_2D took 14026 ticks (14 ms)
DEPTHWISE_CONV_2D took 5087 ticks (5 ms)
CONV_2D took 8200 ticks (8 ms)
DEPTHWISE_CONV_2D took 9880 ticks (9 ms)
CONV_2D took 11456 ticks (11 ms)
DEPTHWISE_CONV_2D took 2475 ticks (2 ms)
CONV_2D took 5792 ticks (5 ms)
DEPTHWISE_CONV_2D took 4792 ticks (4 ms)
CONV_2D took 9076 ticks (9 ms)
DEPTHWISE_CONV_2D took 1204 ticks (1 ms)
CONV_2D took 4724 ticks (4 ms)
DEPTHWISE_CONV_2D took 2290 ticks (2 ms)
CONV_2D took 8170 ticks (8 ms)
DEPTHWISE_CONV_2D took 2290 ticks (2 ms)
CONV_2D took 8170 ticks (8 ms)
DEPTHWISE_CONV_2D took 2290 ticks (2 ms)
CONV_2D took 8171 ticks (8 ms)
DEPTHWISE_CONV_2D took 2290 ticks (2 ms)
CONV_2D took 8171 ticks (8 ms)
DEPTHWISE_CONV_2D took 2290 ticks (2 ms)
CONV_2D took 8170 ticks (8 ms)
DEPTHWISE_CONV_2D took 580 ticks (0 ms)
CONV_2D took 4883 ticks (4 ms)
DEPTHWISE_CONV_2D took 1056 ticks (1 ms)
CONV_2D took 9034 ticks (9 ms)
MEAN took 2378 ticks (2 ms)
FULLY_CONNECTED took 56 ticks (0 ms)
"""

# ---------------------------------------------------------
# 2. 数据解析 (Parsing)
# ---------------------------------------------------------
# 使用正则表达式提取算子名称和Ticks
pattern = re.compile(r"([A-Z_0-9]+)\s+took\s+(\d+)\s+ticks")
matches = pattern.findall(log_data)

# 创建 DataFrame
df = pd.DataFrame(matches, columns=['Operator', 'Ticks'])
df['Ticks'] = df['Ticks'].astype(int)
df['Layer_Index'] = df.index + 1  # 添加层号索引

# ---------------------------------------------------------
# 3. 可视化 (Visualization)
# ---------------------------------------------------------
# 设置风格
sns.set_theme(style="whitegrid")
plt.figure(figsize=(14, 10))

# 图1: Layer-wise Execution (时序图)
plt.subplot(2, 1, 1)
# 使用不同颜色区分算子类型
sns.barplot(
    x='Layer_Index', y='Ticks', hue='Operator', data=df, dodge=False, palette='viridis'
)
plt.title('Layer-wise Execution Profile (Sequential)', fontsize=14, fontweight='bold')
plt.xlabel('Layer Index (Execution Order)', fontsize=12)
plt.ylabel('Execution Time (Ticks)', fontsize=12)
plt.legend(title='Operator Type', bbox_to_anchor=(1.01, 1), loc='upper left')

# 图2: Operator Type Summary (算子汇总条形图)
plt.subplot(2, 2, 3)
op_summary = df.groupby('Operator')['Ticks'].sum().reset_index()
op_summary = op_summary.sort_values(by='Ticks', ascending=False)
sns.barplot(x='Ticks', y='Operator', data=op_summary, palette='magma')
plt.title('Total Time by Operator Type', fontsize=14, fontweight='bold')
plt.xlabel('Total Ticks', fontsize=12)
# 在条形图上添加数值标签
for i, (index, row) in enumerate(op_summary.iterrows()):
    plt.text(row.Ticks, i, f" {row.Ticks}", va='center')

# 图3: Percentage Pie Chart (占比饼图)
plt.subplot(2, 2, 4)
colors = sns.color_palette('pastel')[0 : len(op_summary)]
plt.pie(
    op_summary['Ticks'],
    labels=op_summary['Operator'].tolist(),
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    pctdistance=0.85,
    explode=[0.05] * len(op_summary),
)
# 画一个白圈变成甜甜圈图
centre_circle = mpatches.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Execution Time Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()

# save pdf
plt.savefig("profiling_report.pdf")
plt.show()
