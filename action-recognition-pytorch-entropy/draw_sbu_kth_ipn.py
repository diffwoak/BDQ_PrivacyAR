import matplotlib.pyplot as plt
import numpy as np

# 示例数据
methods = ["Ideal", "Orig. video", "Wu et al.", "BDQ", "Ryoo et al."]
colors = ["gray", "orange", "blue", "red", "brown"]
markers = ["*", "s", "o", "*", "x"]

data = {
    "Actor-Pair Accuracy": [0, 95, 50, 30, 85],
    "Identity Accuracy": [0, 93, 45, 25, 80],
    "Gender Accuracy": [0, 90, 55, 60, 82],
    "Action Accuracy": [100, 90, 80, 85, 75],
    "Actions Accuracy": [100, 90, 80, 85, 75],
    "Hand Gesture Accuracy": [100, 88, 78, 83, 73],
}

# 创建图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 绘制第一个图
axes[0].set_title("Actor-Pair vs Action Accuracy")
for i, method in enumerate(methods):
    axes[0].scatter(data["Actor-Pair Accuracy"][i], data["Action Accuracy"][i], 
                    label=method, color=colors[i], marker=markers[i], s=100)
axes[0].set_xlabel("Actor-Pair Accuracy (%)")
axes[0].set_ylabel("Action Accuracy (%)")
axes[0].set_xlim(-5, 105)
axes[0].set_ylim(-5, 105)
axes[0].grid(True, alpha=0.6)  # 添加网格

# 绘制第二个图
axes[1].set_title("Identity vs Action Accuracy")
for i, method in enumerate(methods):
    axes[1].scatter(data["Identity Accuracy"][i], data["Actions Accuracy"][i], 
                    label=method, color=colors[i], marker=markers[i], s=100)
axes[1].set_xlabel("Identity Accuracy (%)")
axes[1].set_ylabel("Action Accuracy (%)")
axes[1].set_xlim(-5, 105)
axes[1].set_ylim(-5, 105)
axes[1].grid(True, alpha=0.6)  # 添加网格

# 绘制第三个图
axes[2].set_title("Gender vs Hand Gesture Accuracy")
for i, method in enumerate(methods):
    axes[2].scatter(data["Gender Accuracy"][i], data["Hand Gesture Accuracy"][i], 
                    label=method, color=colors[i], marker=markers[i], s=100)
axes[2].set_xlabel("Gender Accuracy (%)")
axes[2].set_ylabel("Hand Gesture Accuracy (%)")
axes[2].set_xlim(-5, 105)
axes[2].set_ylim(-5, 105)
axes[2].grid(True, alpha=0.6)  # 添加网格

# 添加图例
axes[0].legend(loc="lower left")
axes[1].legend(loc="lower left")
axes[2].legend(loc="lower left")

# 保存为图片
plt.tight_layout()
plt.savefig("scatter_plot_fixed_range.png", dpi=300, bbox_inches="tight")
plt.close()  # 关闭图像
