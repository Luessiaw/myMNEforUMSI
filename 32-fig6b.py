# 绘制图 6b
# import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
import gc

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

view_init = [30,-30]
rp_phis = np.array([-50,0])/180*np.pi
rp_theta = 45/180*np.pi
p = np.array([0,10e-9,0])
rps = []
ps = []
for i,rp_phi in enumerate(rp_phis):
    rp = np.array([np.sin(rp_theta)*np.cos(rp_phi),
                np.sin(rp_theta)*np.sin(rp_phi),
                np.cos(rp_theta)])*9e-2
    rps.append(rp)
    p = np.cross(unit_z,rp)
    p = 10e-9*p/np.linalg.norm(p)
    ps.append(p)

# fig6a-1 绘制源
fig1 = vs.plt.figure()
fig1.set_size_inches(10,8)
vz = Visualizer()

fig1,ax1 = vz.create3DAxis(3,fig=fig1,lims=[0.15,0.15,0.15])
ax1.view_init(elev=view_init[0], azim=view_init[1])
ax1.set_axis_off()

vz.showHead(paras.radiusOfHead,3,ax1,alpha=0.1)

for i in range(len(rps)):
    vz.showSource(rps[i],ps[i],ax1,3,
                arrowBottomRadius=0.002,arrowTipRadius=0.005,
                arrowBottomLength=0.02,arrowTipLength=0.01,
                color="red")

# vz.showAxisArrow(ax1,0.11,0.11,0.11,
#                 bottomRadius=0.0005,tipRadius=0.002,
#                 xTipLength=0.01,yTipLength=0.01,zTipLength=0.01)

fig1.savefig("figs/fig6a/6a-1.png",
             dpi=300,bbox_inches='tight', pad_inches=0)
vs.plt.close(fig1)
print("Done")

# fig6a-2 对两个源进行源成像
paras.gridSpacing = 0.3e-2
paras.sourceOnSpheres = (np.linspace(7,9,5,endpoint=True,dtype=np.float32)*1e-2).tolist()
# paras.sourceOnSpheres = [0.09]
paras.externalNoise = 10e-15
paras.dipoleRestrict = True # 限定偶极子的方向
paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                numOfChannelsForDim3=128)
paras3v.theta = 0
paras3s.theta = 0
paras3s.GeoFieldAtRef = 5e-5*(unit_x*np.sin(paras3s.theta)+unit_z*np.cos(paras3s.theta))


for (k,par) in enumerate([paras3s,paras3v]):
    gc.collect()
    sol = Solver(par)
    gc.collect()
    print(f"num of source points: {sol.sourcePoints.shape[1]}")
    points = sol.sourcePoints
    distance = np.sqrt(points[0,:]**2+points[1,:]**2+points[2,:]**2)
    radius = sol.paras.sourceOnSpheres[-1] # 取最外层
    # mask = (distance<radius*1.1)&(distance>radius*0.9)&(points[2,:]>=0)
    mask = (distance<0.15)&(distance>0.06)
    xs = points[0,mask]
    ys = points[1,mask]
    zs = points[2,mask]

    Qs = []

    # 仅源1或源2打开
    for j in range(2):
        rp = rps[j]
        p = ps[j]
        Bm = sol.getBm(rp,p)
        Q = sol.applyInverse(Bm)
        Qs.append(Q)

    # 源1和源2同时打开
    Q = Qs[0] + Qs[1]
    Qs.append(Q)
    for j in range(3):
        QA = sol.getQAmplitute(Qs[j])
        QAm = QA[mask]
        QAm /= QAm.max()
        
        fig2 = vs.plt.figure()
        fig2.set_size_inches(10,8)
        vz = Visualizer()

        fig2,ax2 = vz.create3DAxis(par.dim,fig=fig2,lims=[0.15,0.15,0.15])
        # ax2.set_axis_off()
        ax2.view_init(elev=view_init[0], azim=view_init[1])

        # 重新绘制 box
        ax2.set_axis_off()
        xlim = np.array([-0.1,0.1])
        ylim = np.array([-0.1,0.1])
        # zlim = np.array([-0.01,1.01])
        zlim = np.array([-0.1,0.1])
        # kwargs = {"color":"k","lw":1}
        # # 绘制底面边线
        # ax2.plot(xlim[[0,1]],ylim[[0,0]],zlim[[0,0]],**kwargs)
        # # ax2.plot(xlim[[0,1]],ylim[[1,1]],zlim[[0,0]],**kwargs)
        # # ax2.plot(xlim[[0,0]],ylim[[0,1]],zlim[[0,0]],**kwargs)
        # ax2.plot(xlim[[1,1]],ylim[[0,1]],zlim[[0,0]],**kwargs)
        # # 绘制顶面边线
        # ax2.plot(xlim[[0,1]],ylim[[0,0]],zlim[[1,1]],**kwargs)
        # ax2.plot(xlim[[0,1]],ylim[[1,1]],zlim[[1,1]],**kwargs)
        # ax2.plot(xlim[[0,0]],ylim[[0,1]],zlim[[1,1]],**kwargs)
        # ax2.plot(xlim[[1,1]],ylim[[0,1]],zlim[[1,1]],**kwargs)
        # # 绘制竖线
        # ax2.plot(xlim[[0,0]],ylim[[0,0]],zlim[[0,1]],**kwargs)
        # # ax2.plot(xlim[[0,0]],ylim[[1,1]],zlim[[0,1]],**kwargs)
        # # ax2.plot(xlim[[1,1]],ylim[[0,0]],zlim[[0,1]],**kwargs)
        # ax2.plot(xlim[[1,1]],ylim[[1,1]],zlim[[0,1]],**kwargs)

        ax2.scatter(xs,ys,zs,c=QAm,cmap="Reds",s=45)
        ax2.set(xlim=xlim,ylim=ylim,zlim=zlim)
        ax2.set(xticks=[],yticks=[],zticks=[])
        ax2.set_aspect("equalxy")

        fig2.canvas.draw()
        fig2.savefig(f"figs/fig6a/6a-2-{par.getLabel()}-{j}.png",
                    dpi=300,bbox_inches='tight',pad_inches=0)
        vs.plt.close(fig2)

print("Done.")
