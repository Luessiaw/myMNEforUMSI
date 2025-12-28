import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

def getErr(filePath):
    data = np.load(filePath)
    err = data["errs"]
    err *= 1e2
    vmin = np.min(err)
    vmax = np.max(err)
    return err,vmin,vmax

def drawHarfCircle(ax:plt.Axes):
    # 生成半圆的参数点（上半圆）
    radius = 10
    theta = np.linspace(0, np.pi, 100)  # 角度从0到π
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    ax.plot(x, y, 'k-', linewidth=6)  # 绘制半圆曲线

oris = ["x","z","xAz"]

labels = []
errs = []
vmins = []
vmaxs = []
for ori in oris:
    folder = f"figs/regularized-deadzone/z-x/geo-{ori}"

    for t in ["3sB","3sW","3v"]:
        filePath = os.path.join(folder,f"z-x-{t}.npz")
        err,vmin,vmax = getErr(filePath)
        errs.append(err)
        vmins.append(vmin)
        vmaxs.append(vmax)
        labels.append(f"{ori}-{t}")        

vmin = np.min(np.array(vmins))
vmax = np.max(np.array(vmaxs))

x = np.linspace(-10,10,50)
z = np.linspace(0,10,25)
X,Z = np.meshgrid(x, z)

xi = np.linspace(x.min(), x.max(), 100)
zi = np.linspace(z.min(), z.max(), 100)
Xi, Zi = np.meshgrid(xi, zi)

Ri = np.sqrt(Xi**2 + Zi**2)
mask = Ri>10
for i in range(len(labels)):
    label = labels[i]
    err = errs[i]
    erri = griddata((X.flatten(), Z.flatten()), err.flatten(), (Xi, Zi), method='cubic')
    erri[mask] = np.nan
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    im = ax.pcolormesh(Xi,Zi,erri,shading="auto",vmin=vmin,vmax=vmax)
    drawHarfCircle(ax)
    ax.set_aspect(1)
    ax.set_title(label)
    ax.set_xlabel("x (cm)")   
    ax.set_ylabel("z (cm)")
    
    # 获取主图的坐标范围
    ax_pos = ax.get_position()
    cbar_height = ax_pos.height  # 使用主图高度
    # 创建与主图等高的colorbar
    cax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, cbar_height])
    cb = plt.colorbar(im, cax=cax,label="Localization Error (cm)")

    fig.savefig(f"figs/regularized-deadzone/z-x/total/{label}")
    plt.close(fig)

# plt.show()

