import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(1.2, 4))

norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = mpl.cm.viridis

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 关键：避免警告

cbar = fig.colorbar(sm, cax=ax)

plt.show()
