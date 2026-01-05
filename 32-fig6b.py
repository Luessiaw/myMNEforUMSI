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
radius = 0.09
ds = np.array([3.5,4.5,5.5])/1e2
rp_theta = 45/180*np.pi
rp_phis = -np.array([2*np.arcsin(d/radius/2/np.sin(rp_theta)) for d in ds])


# fig6a-2 对两个源进行源成像
paras.gridSpacing = 0.3e-2
# paras.sourceOnSpheres = (np.linspace(7,9,5,endpoint=True,dtype=np.float32)*1e-2).tolist()
paras.sourceOnSpheres = [0.09]
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
    mask = (distance<0.092)&(distance>0.089)&(points[2,:]>=0)
    xs = points[0,mask]
    ys = points[1,mask]
    zs = points[2,mask]

    for rp_phi in rp_phis:
        d = 9*np.sin(rp_theta)*np.sin(-rp_phi/2)*2
        print(f"Distance: {d:.1f} cm.")
        
        rps = []
        ps = []
        for rp_phi2 in [0,rp_phi]:
            rp = np.array([np.sin(rp_theta)*np.cos(rp_phi2),
                        np.sin(rp_theta)*np.sin(rp_phi2),
                        np.cos(rp_theta)])*9e-2
            rps.append(rp)
            p = np.cross(unit_z,rp)
            p = 10e-9*p/np.linalg.norm(p)
            ps.append(p)

        Bm = 0
        for j in range(2):
            Bm += sol.getBm(rps[j],ps[j],not j)
        Q = sol.applyInverse(Bm)
        QA = sol.getQAmplitute(Q)
        QAm = QA[mask]
        QAm /= QAm.max()
        # X,Y,V = interpXYs(xs,ys,QAm,nx=200,ny=200,xlim=[xs.min(),xs.max()],ylim=[ys.min(),ys.max()],method="cubic")
        radiusOfTheta = radius*np.sin(rp_theta)
        
        fig2 = vs.plt.figure()
        fig2.set_size_inches(10,8)
        vz = Visualizer()
        fig2,ax2 = vz.create3DAxis(par.dim,fig=fig2,lims=[0.15,0.15,0.15])
        ax2.set_proj_type('ortho')
        ax2.view_init(elev=view_init[0], azim=view_init[1])

        xlim = np.array([-0.1,0.1])
        ylim = np.array([-0.1,0.1])
        zlim = np.array([-0.01,1.01])

        # norm = vs.plt.Normalize(vmin=0, vmax=1)  # 归一化到 [0,1]
        # cmap = vs.cm.Reds  # 选择颜色映射（如 'viridis', 'jet', 'plasma'）
        # colors = cmap(norm(V))
        # ax2.plot_surface(X,Y,V,facecolors=colors,edgecolor="none",shade=False)
        ax2.scatter(xs,ys,QAm,c=QAm,cmap="Reds",s=400)
        ax2.set(xlim=xlim,ylim=ylim,zlim=zlim)

        # ax2.grid(True, linewidth=5)
        ax2.tick_params(axis='x', labelbottom=False)
        ax2.tick_params(axis='y', labelleft=False)
        ax2.tick_params(axis='z', labelleft=False)

        ax2.set_aspect("equalxy")

        fig2.canvas.draw()
        fig2.savefig(f"figs/fig6b/6b-{par.getLabel()}-{d:.1f}cm.png",
                    dpi=300,bbox_inches='tight',pad_inches=0)
        vs.plt.close(fig2)

print("Done.")
