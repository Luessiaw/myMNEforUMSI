import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(0.6, 4))

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = mpl.cm.bwr

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 关键：避免警告

cbar = fig.colorbar(sm, cax=ax)

cbar.set_ticks([])          # 去掉刻度
cbar.outline.set_visible(False)  # 去掉 colorbar 外框

# 去掉 Axes 自身的边框
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_yticks([])           # 冗余但安全
ax.set_xticks([])

fig.savefig("figs/fig6c/cb-bwr.png")

plt.show()
