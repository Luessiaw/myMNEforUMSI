# 地磁场方向的误差
# import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

# paras.dim
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 100e-9
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

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = False
# paras.labelPostfix

varName = "effective-width"
refreshMode = 3 # 不需要重新计算 L,W
xs = np.linspace(0,200,11)
sigmas = xs*1e-15

saveFolder = "data"

def getMoment(p:Paras,sigma:float):
    sol = Solver(p)
    trial = sol.singleTrial()
    Bm = trial.Bm
    rs = sol.sensorPoints
    weight = (Bm**2+sigma**2)/(np.sum(Bm**2+sigma**2))
    r_mean = np.einsum("j,ij->i",weight,rs)
    rir = np.sum((rs.transpose() - r_mean).transpose()**2,axis=0)
    width = np.einsum("i,i->",weight,rir)
    width = np.sqrt(width)
    return Bm,rs,width

rp_phis = np.array([0,30])/180*np.pi
rp_theta = 30/180*np.pi
rps = []
ps = []
strengths = [10,10]
for i,phi in enumerate(rp_phis):
    rp = np.array([np.sin(rp_theta)*np.cos(phi),
                   np.sin(rp_theta)*np.sin(phi),
                   np.cos(rp_theta)])*8e-2
    rps.append(rp)
    p = np.cross(unit_z,rp)
    p = strengths[i]*1e-9*p/np.linalg.norm(p)
    ps.append(p)

p = np.array([0,10e-9,0])
# sources = [(rp1,p),(rp2,p)]

# 格点间距为 3 或 4 mm 时可看出UMSI与SMSI的差距。
paras.gridSpacing = 0.3e-2
paras.sourceOnSpheres = (np.linspace(7,9,5,endpoint=True,dtype=np.float32)*1e-2).tolist()
# paras.gridSpacing = 1e-2
# paras.sourceOnSpheres = (np.linspace(7,9,3,endpoint=True)*1e-2).tolist()
paras.externalNoise = 10e-15


# # 在一个源盘上
# xs,ys = disk_fibonacci(4000)
# xs *= 5/1e2
# ys *= 5/1e2
# zs = np.zeros_like(xs) + 4*np.sqrt(3)/1e2
# paras.sourcePoints = np.array([xs,ys,zs])

# # 在半径为 8cm 的球面上
# M = 10000
# points = []
# for r in [7,8,9]:
#     points1 = fibonacci_sphere(M)
#     points.append(np.array(points1).transpose()*r/1e2)
# paras.sourcePoints = np.hstack(points)

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                numOfChannelsForDim3=128)

paras3v.theta = 0

paras3s.theta = 0
paras3s.GeoFieldAtRef = 5e-5*(unit_x*np.sin(paras3s.theta)+unit_z*np.cos(paras3s.theta))

task = 1
task = 2
task = 3
print(f"task {task}.")
t = time.time()
colors = ["red","blue","green"]
for par in [paras3s,paras3v]:
    # rp,p = par.fixDipole

    fig = vs.plt.figure()
    fig.set_size_inches(10,8)
    vz = Visualizer()

    # 绘制头部
    fig,ax = vz.create3DAxis(par.dim,fig=fig,lims=[0.15,0.15,0.15])
    ax.set_axis_off()

    if task == 1:
        # 绘制源
        vz.showHead(par.radiusOfHead,par.dim,ax,alpha=0.1)
        for i in range(len(rps)):
            vz.showSource(rps[i],ps[i],ax,par.dim,
                        arrowBottomRadius=0.001,arrowTipRadius=0.0025,
                        arrowBottomLength=0.01,arrowTipLength=0.005,
                        color=colors[i%len(colors)])
        vz.showAxisArrow(ax,0.11,0.11,0.11,
                        bottomRadius=0.0005,tipRadius=0.002,
                        xTipLength=0.01,yTipLength=0.01,zTipLength=0.01)
        png_name = f"source"
    elif task ==2:
        # 绘制测量值
        sol = Solver(par)
        vz.showMeasuredBForMultiDipoles(ax,sol,rps,ps,vmin=-8e-13,vmax=8e-13,num=3000,alpha=1,printExtrim=True,cmap="coolwarm")
        png_name = f"measurement"

    elif task == 3:
        # 绘制源定位结果
        sol = Solver(par)
        print(f"num of source points: {sol.sourcePoints.shape[1]}")
        # trial = sol.singleTrial()
        Bm = 0
        for i in range(len(rps)):
            vz.showSource(rps[i],ps[i],ax,par.dim,
                        arrowBottomRadius=0.001,arrowTipRadius=0.0025,
                        arrowBottomLength=0.01,arrowTipLength=0.005,
                        color=colors[i%len(colors)])
        for i in range(len(rps)):
            Bm += sol.getBm(rps[i],ps[i],not i) # not i 表示只计算一次噪声
        Q = sol.applyInverse(Bm)

        vz.showHead(par.radiusOfHead,par.dim,ax,alpha=0.05)
        vz.showImagingResult(sol,Q,ax,10,alpha=1,cmap="Reds")
        png_name = f"result"

    fig.canvas.draw()
    fig.savefig(f"figs/双源分辨/{str(int(t))[5:]}-{png_name}-{par.getLabel()}.png",
                dpi=300,bbox_inches='tight', 
                pad_inches=0,
                transparent=True)
    vs.plt.close(fig)
    
print("Done.")
