# 看起来 UMSL 比 SMSL 更加局域，那么它对于源成像是否有利？
# import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
from matplotlib.gridspec import GridSpec

paras = Paras()

paras.dim = 3
paras.radiusOfHead = 100e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
# paras.dipoleRadiusRange = np.array([8e-2,10e-2]) # 位置固定
# paras.dipoleThetaRange = np.array([0,np.pi/3]) 
# paras.dipolePhiRange = np.array([0,0]) # 位置固定

# paras.sensorType
# paras.numOfChannels = 64
paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 0e-15
paras.externalNoise = 0
paras.considerDeadZone = False
paras.deadZoneType = "best" # best, worst, random
paras.axisAngleError = 0 
paras.considerRegistrate = False
paras.registrateType = "best"
paras.registrateError = 0

# paras.GeoRefPos = origin
# # theta = np.pi/2
# theta = 0 # 主磁场方向沿 z 轴
# paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
# paras.GeoFieldGradientTensor = np.zeros((3,3))
# paras.GeoFieldGradientKnown = True

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None
# paras.labelPostfix

varName = "UMSL局域性"
saveFolder = os.path.join("figs",f"{varName}")
print(f"Save folder: {saveFolder}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)

solvers:list[Solver] = []

pv = deepcopy(paras)
pv.sensorType = "vector"
pv.labelPostfix = ""
solvers.append(Solver(pv))

ps = deepcopy(paras)
ps.sensorType = "scalar"
ps.GeoRefPos = origin
ps.GeoFieldAtRef = 5e-5*unit_x
ps.GeoFieldGradientTensor = np.zeros((3,3))
ps.GeoFieldGradientKnown = True
solvers.append(Solver(ps))

def getTheoBm(solver:Solver,rp:np.ndarray,p:np.ndarray,num=100):
    '''rotAxis: 偏转轴，整体偏转。
    rotAng:偏转角度，弧度。'''
    rS = solver.paras.radiusOfSensorShell
    thetas = np.linspace(0,np.pi/2,num)
    phis = np.linspace(0,2*np.pi,num)
    x = rS*np.linspace(-1,1,num)
    y = rS*np.linspace(-1,1,num)
    xv,yv = np.meshgrid(x,y,indexing="ij")
    inside = xv**2 + yv**2 <= rS**2
    zv = np.zeros_like(xv)
    zv[inside] = np.sqrt(rS**2 - xv[inside]**2 - yv[inside]**2)
    zv[~inside] = np.nan
    # zv = np.sqrt(rS**2-xv**2-yv**2)
    B = np.zeros((num,num))

    for i in range(num):
        for j in range(num):
            r = np.array([xv[i,j],yv[i,j],zv[i,j]])
            R = r - rp
            nr = np.linalg.norm(r)
            nR = np.linalg.norm(R)
            F = nR*(nr*nR+np.vdot(r,R))
            nablaF = (nR**2/nr+nR)*r + (np.vdot(r,R)/nR+nR+2*nr)*R
            q = np.cross(p,rp)

            if solver.paras.sensorType == "scalar":
                Bgeo = solver.paras.GeoFieldAtRef + np.dot(solver.paras.GeoFieldGradientTensor,r)
                nSensor = Bgeo/np.linalg.norm(Bgeo)
            elif solver.paras.sensorType == "vector":
                nSensor = r/np.linalg.norm(r)

            Bi = k0*(q/F-np.vdot(q,r)*nablaF/F**2)
            Bij = np.vdot(nSensor,Bi)
            B[i,j] = Bij

    return xv,yv,B


BmMax = 0
BmMin = np.inf
rp = np.array([4,0,4*np.sqrt(3)])/1e2
p = np.array([0,100e-9,0])

logging.info("Calculating...")
results = []
resultls = []
for s in solvers:
    xv,yv,Bm = getTheoBm(s,rp,p)
    BmMax = max(BmMax,np.nanmax(Bm))
    BmMin = min(BmMin,np.nanmin(Bm))
    results.append([xv,yv,Bm])

    Bml = Bm[:,50]
    xv = xv[:,50]
    resultls.append([xv,Bml])

logging.info(f"B max: {BmMax*1e12:.2f} pT. B min: {BmMin*1e12:.2f} pT.")
logging.info("Calculations done.")
logging.info("Drawing...")

fig = vs.plt.figure(figsize=(15,3.5))
gs = GridSpec(1, 4, width_ratios=[0.05,1, 1, 1.5],wspace=0.3)
axLabels = ["Shielded, radial","Unshielded, x"]
axCap = ["(a)","(b)","(c)"]
for (i,(xv,yv,Bm)) in enumerate(results):
    # ax = fig.add_axes([0.1+0.5*i,0.4,0.4,0.4])
    ax = fig.add_subplot(gs[0,i+1])
    norm = vs.plt.Normalize(vmin=BmMin*1e12, vmax=BmMax*1e12)  # 归一化到 [0,1]
    cmap = vs.cm.rainbow  # 选择颜色映射（如 'viridis', 'jet', 'plasma'）
    colors = cmap(norm(Bm)) 
    pcm = ax.pcolormesh(xv*1e2,yv*1e2,Bm*1e12,shading="auto",cmap=cmap,norm=norm)
    ax.set_xlabel("x (cm)")
 
    if not i:
        ax.set_ylabel("y (cm)")
        # ax.spines["right"].set_visible(False)
    else:
    #     # ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False, left=False)
    #     cbar = vs.plt.colorbar(pcm, ax=ax)
    #     cbar.set_label("B (pT)", rotation=270, labelpad=15)
    ax.set_title(axLabels[i])
    ax.set_aspect(1)
    ax.text(0.05,0.9,axCap[i],transform=ax.transAxes)

ax = fig.add_subplot(gs[0,0])
cbar = fig.colorbar(pcm, cax=ax, orientation='vertical', ticklocation='left')
cbar.set_label("B (pT)", rotation=90, labelpad=15)

ax = fig.add_subplot(gs[0,3])
for (i,(xl,Bml)) in enumerate(resultls):    
    ax.plot(xl*1e2,Bml*1e12,label=axLabels[i])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("B (pT)")
ax.legend(loc='lower left')
ax.text(0.05,0.9,axCap[2],transform=ax.transAxes)

# ax = fig.add_subplot(gs[1,1])
# for (i,(xl,Bml)) in enumerate(resultls):    
#     ax.plot(xl*1e2,Bml*1e12,label=axLabels[i])
#     ax.set_xlabel("x (cm)")
#     ax.set_ylabel("B (pT)")
# ax.legend()

vs.plt.tight_layout()

fig.savefig(os.path.join(saveFolder,f"BrBx.jpg"),dpi=600)
vs.plt.close(fig)

logging.info("done.")