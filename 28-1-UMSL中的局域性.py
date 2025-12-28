# 看起来 UMSL 比 SMSL 更加局域，那么它对于源成像是否有利？
# import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

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
ps.GeoFieldGradientTensor = np.zeros((3,3))
ps.GeoFieldGradientKnown = True
units = [unit_x,unit_y,unit_z]
fix = ["x","y","z"]
for fix in ["x","y","z"]:
    psf = deepcopy(ps)
    psf.GeoFieldAtRef = 5e-5*eval(f"unit_{fix}")
    # logging.info(f"{ps.GeoFieldAtRef}")
    psf.labelPostfix = fix
    solvers.append(Solver(psf))

def getTheoBm(solver:Solver,rp:np.ndarray,p:np.ndarray,num=100):
    '''rotAxis: 偏转轴，整体偏转。
    rotAng:偏转角度，弧度。'''
    rS = solver.paras.radiusOfSensorShell
    thetas = np.linspace(0,np.pi/2,num)
    phis = np.linspace(0,2*np.pi,num)
    x = rS*np.linspace(-1,1,num)
    y = rS*np.linspace(-1,1,num)
    xv,yv = np.meshgrid(x,y,indexing="ij")
    # mask = xv**2+yv**2 <= rS**2
    # xv = xv[mask]
    # yv = yv[mask]
    zv = np.sqrt(rS**2-xv**2-yv**2)
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

def plotMeas2D(xv:np.ndarray,yv:np.ndarray,Bm:np.ndarray,ax:vs.plt.Axes,cb:False):
    # norm = vs.plt.Normalize(vmin=Bm.min(), vmax=Bm.max())  # 归一化到 [0,1]
    # cmap = vs.cm.rainbow  # 选择颜色映射（如 'viridis', 'jet', 'plasma'）
    # colors = cmap(norm(Bm)) 
    pcm = ax.pcolormesh(xv*1e2,yv*1e2,Bm*1e12,shading="auto")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect(1)
    if cb:
        cbar = vs.plt.colorbar(pcm, ax=ax)
        cbar.set_label("B (pT)", rotation=270, labelpad=15)

rp_thetas = np.array([0,30,60,90])/180*np.pi
nrows,ncols = map(len,[rp_thetas,solvers])
BmMax = 0
BmMin = np.inf
logging.info("Calculating...")

results = []
for i,rp_theta in enumerate(rp_thetas):
    logging.info(f"Point number: {i+1}")
    line = []
    rp = np.array([8*np.sin(rp_theta),0,8*np.cos(rp_theta)])/1e2
    p = np.array([0,100e-9,0])

    for (j,s) in enumerate(solvers):
        logging.info(f"{s.paras.getLabel()}")

        xv,yv,Bm = getTheoBm(s,rp,p)
        BmMax = max(BmMax,np.max(Bm))
        BmMin = min(BmMin,np.min(Bm))
        line.append([xv,yv,Bm])

    results.append(line)
logging.info("Calculations done.")
logging.info("Drawing...")

fig = vs.plt.figure(figsize=(15,12))
for (i,line) in enumerate(results):
    logging.info(f"Point number: {i+1}")
    for (j,res) in enumerate(line):
        logging.info(f"{solvers[j].paras.getLabel()}")
        xv,yv,Bm = res
        ax = fig.add_subplot(nrows,ncols,ncols*i+j+1)
        plotMeas2D(xv,yv,Bm,ax,True)
        if not i:
            ax.set_title(solvers[j].paras.getLabel())

    fig.savefig(os.path.join(saveFolder,f"cb-UvsSMSL.jpg"),dpi=600)
    vs.plt.close(fig)

logging.info("done.")