import matplotlib.pyplot as plt
import numpy as np

# 示例数据
methods = ["Orig. video", "B", "D", "Q", "B+D", "B+Q", "D+Q","Orig. BDQ","New. BDQ"]
colors = ["blue", "gray", "brown", "green", "pink", "magenta", "purple", "orange", "red"]
markers = ["s", "v", "p", "h", "1", "d", "H", "D", "*"]

data = {
    "Actor-Pair Accuracy": [0, 95, 50, 30, 85, 90, 54, 50, 75],
    "Action Accuracy": [100, 90, 80, 85, 75, 80, 70, 35, 45],
}

# 创建图
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

# 绘制第一个图
# axes.set_title("Actor-Pair vs Action Accuracy")
for i, method in enumerate(methods):
    axes.scatter(data["Actor-Pair Accuracy"][i], data["Action Accuracy"][i], 
                    label=method, color=colors[i], marker=markers[i], s=30)
axes.set_xlabel("Actor-Pair Accuracy (%)")
axes.set_ylabel("Action Accuracy (%)")
axes.set_xlim(-5, 105)
axes.set_ylim(-5, 105)
axes.grid(True, alpha=0.6)  # 添加网格


# 添加图例
axes.legend(loc="lower left")

# 保存为图片
plt.tight_layout()
plt.savefig("scatter_plot_fixed.png", dpi=300, bbox_inches="tight")
plt.close()  # 关闭图像
