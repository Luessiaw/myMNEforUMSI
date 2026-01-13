import matplotlib
# matplotlib.use("Agg") 

import numpy as np
from myMNE import *
import glob
from scipy.signal import butter, sosfiltfilt
from PIL import Image
import os


def butter_bandpass_sos(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass',
                 output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass_sos(lowcut, highcut, fs, order)
    y = sosfiltfilt(sos, data)
    return y

paras = Paras()
# paras.dim
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 1000e-9
paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 位置固定
paras.dipoleThetaRange = np.array([0,0]) 
paras.dipolePhiRange = np.array([0,0])

# paras.sensorType
paras.numOfChannels = 128
paras.radiusOfSensorShell = 11e-2
# paras.intrisicNoise = 10e-15
paras.intrisicNoise = 0
paras.externalNoise = 0
paras.considerDeadZone = False
paras.deadZoneType = "best" # best, worst, random
paras.axisAngleError = 0 
paras.considerRegistrate = False
paras.registrateType = "best"
paras.registrateError = 0

paras.GeoRefPos = origin
# theta = np.pi/2
theta = 0 # 主磁场方向沿 z 轴
paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
# paras.GeoFieldAtRef = None
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = False

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 1
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = False
# paras.labelPostfix

saveFolder = "data"

# 位于固定位置的源
rp = np.array([4.5,0,4.5*np.sqrt(3)])/1e2
p = np.array([0,100,0])/1e9
paras.fixDipole = (rp,p)

# 只考虑地磁环境，无噪声
paras.gridSpacing = 0.3e-2
paras.sourceOnSpheres = [0.09]

paras.dim = 3
paras.sensorType = "scalar"
paras.externalNoise = 100e-15 # 100fT/rtHz 的噪声
paras.intrisicNoise = 0

# 产生初始测量值分布
sol = Solver(paras)
Bm = sol.getBm(rp,p,False)

# 源信号：20Hz 正弦信号, 采样率 4096, 时长 10 s

fs = 4096
S  = 0.1          # 1/sqrt(Hz)
sigma = S * np.sqrt(fs)
最终信号 = []
每段时长 = 10
N = fs * 每段时长
for k in range(128):
    x = sigma * np.random.randn(N)
    最终信号.append(x)
noise = np.array(最终信号)

noise = noise[:paras.numOfChannels]

N = noise.shape[1]
fs = 4096
T = N/fs
t = np.arange(N)/fs
freq = 20
q = np.sin(2*np.pi*freq*t)
B:np.ndarray = Bm[:, None] * q[None, :]
B *= 1e12
Bnoised = B + noise

# 对信号进行滤波
fl = 19
fh = 21
Bfiltered = butter_bandpass_filter(Bnoised,fl,fh,fs,5)

t1 = 0
t2 = 0.1
t_mask = (t>=t1) & (t<=t2)
idt = np.where(t_mask)[0]
idt = idt[::16]

# 开始绘图
view_init = [30,-30]
vz = Visualizer()

# 真实源
fig1 = vs.plt.figure()
fig1.set_size_inches(10,8)
fig1,ax1 = vz.create3DAxis(3,fig=fig1,lims=[0.15,0.15,0.15])
ax1.view_init(elev=view_init[0], azim=view_init[1])

# 测量值
fig2 = vs.plt.figure()
fig2.set_size_inches(10,8)
ax2 = fig2.add_axes([0.1,0.1,0.7,0.8])
cax = fig2.add_axes([0.85,0.1, 0.05, 0.8]) 

# 成像结果
fig3 = vs.plt.figure()
fig3.set_size_inches(10,8)
fig3,ax3 = vz.create3DAxis(3,fig=fig3,lims=[0.15,0.15,0.15])
ax3.view_init(elev=view_init[0], azim=view_init[1])

# 探头坐标
rxs = sol.sensorPoints[0,:]*1e2
rys = sol.sensorPoints[1,:]*1e2
# 测量值范围
Bmax = np.max(Bfiltered[:,idt])
Bmin = np.min(Bfiltered[:,idt])
dB = Bmax - Bmin
Bmax += 0.1*dB
Bmin -= 0.1*dB

# 格点坐标，用于绘制成像结果
xs = sol.sourcePoints[0,:]*1e2
ys = sol.sourcePoints[1,:]*1e2
zs = sol.sourcePoints[2,:]*1e2
# 坐标轴范围
xlim = np.array([-0.1,0.1])*1e2
ylim = np.array([-0.1,0.1])*1e2
zlim = np.array([-0.1,0.1])*1e2

for i in tqdm.tqdm(idt):
    tick = f"{t[i]*1000:.1f}"

    # 绘制真实源位置
    ax1.cla()
    ax1.set_axis_off()
    vz.showHead(paras.radiusOfHead*1e2,paras.dim,ax1,alpha=0.1)
    amp = np.linalg.norm(q[i]) # 最大振幅为 1
    ax1.scatter(rp[0]*1e2,rp[1]*1e2,rp[2]*1e2, c=amp,cmap="Reds",s=45,vmin=0,vmax=1)
    ax1.set(xlim=xlim,ylim=ylim,zlim=zlim)
    ax1.set(xticks=[],yticks=[],zticks=[])
    ax1.set_aspect("equal")
    ax1.text2D(0.5, 0.85, f"t = {tick} ms",
          transform=ax1.transAxes,
          ha='center',fontsize=16)


    fig1.savefig(f"figs/源定位动图/真实源/{tick}ms.jpeg",dpi=72,
                bbox_inches='tight',pad_inches=0
                )

    # 绘制测量值分布
    B_mask = Bfiltered[:,i]
    ax2.cla()
    s = ax2.scatter(rxs,rys,c=B_mask,cmap="coolwarm",s=45,vmin=Bmin,vmax=Bmax)
    ax2.set(xlim=[-12,12],ylim=[-12,12])
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel("y (cm)")
    ax2.set_aspect("equal")
    fig2.colorbar(s,cax=cax,label="Signal (pT)")
    ax2.set_title(f"t = {tick} ms")

    fig2.savefig(f"figs/源定位动图/测量值/{tick}ms.jpeg",dpi=72,
                # bbox_inches='tight',pad_inches=0
                )

  
    # 进行源定位
    Q = sol.W @ B_mask
    powers = Q[:sol.numOfSourcePoints]**2
    if not paras.dipoleRestrict:
        powers += Q[sol.numOfSourcePoints:]**2
    # 归一化后将低于阈值的设为零
    powersMax = np.max(powers,axis=0)
    powersNorm = powers/powersMax
    powers[powersNorm<paras.threshold] = 0

    QA = sol.getQAmplitute(Q)
    QA /= 455

    # 绘制成像结果
    ax3.cla()    
    ax3.set_axis_off()
    ax3.scatter(xs,ys,zs, c=QA,cmap="Reds",s=45,vmin=0,vmax=1)
    ax3.set(xlim=xlim,ylim=ylim,zlim=zlim)
    ax3.set(xticks=[],yticks=[],zticks=[])
    ax3.set_aspect("equal")
    ax3.text2D(0.5, 0.85, f"t = {tick} ms",
          transform=ax3.transAxes,
          ha='center',fontsize=16)

    fig3.savefig(f"figs/源定位动图/成像结果/{tick}ms.jpeg",dpi=72,
                bbox_inches='tight',pad_inches=0
                )

print()

# 读取图片并制作动图
folders = ["真实源","测量值","成像结果"]
for folder in folders:
    images = []
    filelist = sorted(glob.glob(f"figs/源定位动图/{folder}/*.jpeg"),
                      key=lambda x:float(x.split("\\")[-1][:-5].split("ms")[0]))
    for filename in filelist:
        img = Image.open(filename)
        images.append(img)
    gif_path = f"figs/源定位动图/{folder}.gif"
    images[0].save(gif_path,
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)