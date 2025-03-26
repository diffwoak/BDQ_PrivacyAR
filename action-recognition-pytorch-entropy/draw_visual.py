import matplotlib.pyplot as plt
import numpy as np

# 创建画布与子图布局
fig = plt.figure(figsize=(12, 6), dpi=300)  # 12英寸宽×6英寸高，300dpi
gs = fig.add_gridspec(
    nrows=2, ncols=4,  # 2行4列
    left=0.12, right=0.92,  # 右侧留白用于颜色条
    top=0.88, bottom=0.15,  # 顶部/底部留白
    wspace=0.25, hspace=0.35  # 子图水平/垂直间距
)

# 生成示例数据
x = np.linspace(0, 2*np.pi, 100)
data = [np.sin(x + i*0.5) for i in range(8)]

# 全局样式设置
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8
})

# 绘制子图
axes = []
for i in range(8):
    row = i // 4
    col = i % 4
    ax = fig.add_subplot(gs[row, col])
    
    # 绘制示例曲线
    ax.plot(x, data[i], color='#2ca02c', linewidth=1.5)
    
    # 隐藏内部子图的冗余标签
    if col != 0:
        ax.set_ylabel('')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    if row != 1:
        ax.set_xlabel('')
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # 添加子图标题 (A1, A2,... B1, B2...)
    ax.set_title(f"{chr(65+row)}{col+1}", loc='left', weight='bold', pad=10)
    
    axes.append(ax)

# 添加全局行列标签
fig.text(
    x=0.02, y=0.5, 
    s='Global Y Label', 
    rotation=90, 
    va='center', 
    ha='center', 
    fontsize=12,
    weight='bold'
)

fig.text(
    x=0.5, y=0.95, 
    s='Global X Label', 
    va='center', 
    ha='center', 
    fontsize=12,
    weight='bold'
)

# 保存输出
plt.savefig('multi_panel.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
