# 绘制图 6c
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
parass = [paras3v]
geo_thetas = [0,30,60,90]
for geo_theta in  geo_thetas:
    paras3s2 = deepcopy(paras3s)
    geo_thetaPi = geo_theta/180*np.pi
    paras3s2.GeoFieldAtRef = 5e-5*(unit_x*np.sin(geo_thetaPi)+unit_z*np.cos(geo_thetaPi))
    parass.append(paras3s2)

def get_eff_width(rs:np.ndarray,K:np.ndarray) -> np.ndarray:
    '''
    对每个点，获取峰高
    rs: (3, N)
    K : (N, N)
    return: (N,)
    '''
    W = K**2
    W /= W.sum(axis=0, keepdims=True)
    r_mean = rs @ W
    rs2 = (rs**2) @ W
    rmean2 = (r_mean**2).sum(axis=0)
    widths = np.sqrt(rs2.sum(axis=0) - rmean2)
    return widths        

for (k,par) in enumerate(parass):
    gc.collect()
    sol = Solver(par)
    gc.collect()
    print(f"num of source points: {sol.sourcePoints.shape[1]}")
    points = sol.sourcePoints
    distance = np.sqrt(points[0,:]**2+points[1,:]**2+points[2,:]**2)
    mask = (distance<0.092)&(distance>0.089)&(points[2,:]>=0)
    rs = points[:,mask]
    
    K = sol.W @ sol.L
    widths = get_eff_width(points,K)
    widths = widths[mask]
    if k:
        print(f"theta = {geo_thetas[k-1]}.", end=" ")
    print(f"max width: {widths.max()*100:.1f}, min width: {widths.min()*100:.1f}, mean width: {widths.min()*100:.2f}")
    
    fig2 = vs.plt.figure()
    fig2.set_size_inches(10,8)
    vz = Visualizer()
    fig2,ax2 = vz.create3DAxis(par.dim,fig=fig2,lims=[0.15,0.15,0.15])
    ax2.set_proj_type('ortho')
    ax2.view_init(elev=view_init[0], azim=view_init[1])

    xlim = np.array([-0.12,0.12])
    ylim = np.array([-0.12,0.12])
    zlim = np.array([-0.01,0.12])
    
    ax2.scatter(rs[0,:],rs[1,:],rs[2,:],c=widths,cmap="bwr",s=100,vmax=0.08,vmin=0.04)
    ax2.set(xlim=xlim,ylim=ylim,zlim=zlim)

    # ax2.grid(True, linewidth=5)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(axis='y', labelleft=False)
    ax2.tick_params(axis='z', labelleft=False)

    ax2.set_aspect("equal")
    ax2.set_axis_off()

    fig2.canvas.draw()
    if k:
        pre = f"-{geo_thetas[k-1]}"
    else:
        pre = ""
    fig2.savefig(f"figs/fig6c/6c-{par.getLabel()}{pre}.png",
                dpi=300,bbox_inches='tight',pad_inches=0)
    vs.plt.close(fig2)


print("Done.")
