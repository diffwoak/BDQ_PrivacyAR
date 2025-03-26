import matplotlib.pyplot as plt
import numpy as np

# 示例数据
methods = ["B", "D", "Q", "BD", "BQ", "D+s", "DQ", "BD+s", "DQ+s", "BDQ", "BDQ+s"]
colors = ["#1f77b4", "#7f7f7f", "#8c564b", "#2ca02c", "#e377c2",
          "#d62728", "#9467bd", "#ff7f0e", "#bcbd22", "#17becf", "#393b79"]
markers = ["s", "v", "p", "h", "1", "d", "H", "D", "*", "X", "P"]

data = {
    "Actor-Pair Accuracy": [100, 97.2973, 98.6487, 90.5405, 98.6487, 75.6757, 60.8108, 78.3784, 51.3514, 32.4324,29.7297],
    "Action Accuracy": [89.1892, 93.2432, 89.1892, 90.5405, 85.1351, 87.8378, 78.3784, 85.1351, 78.3784, 72.9730,82.4324],
}

# 创建图
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

# 绘制第一个图
# axes.set_title("Actor-Pair vs Action Accuracy")
for i, method in enumerate(methods):
    axes.scatter(data["Actor-Pair Accuracy"][i], data["Action Accuracy"][i], 
                    label=method, color=colors[i], marker=markers[i], s=20)
axes.set_xlabel("Actor-Pair Accuracy (%)",fontsize=10, labelpad=10)
axes.set_ylabel("Action Accuracy (%)",fontsize=10, labelpad=10)
axes.set_xlim(-5, 105)
axes.set_ylim(-5, 105)
axes.grid(True, alpha=0.6)  # 添加网格


# 添加图例
axes.legend(loc="lower left",fontsize=8)

# 保存为图片
plt.tight_layout()
plt.savefig("scatter_plot_fixed.png", dpi=300, bbox_inches="tight")
plt.close()  # 关闭图像
