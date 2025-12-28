from myMNE import *
from matplotlib.gridspec import GridSpec

varName = "UMSL局域性"
saveFolder = os.path.join("figs",f"{varName}")
logging.info(f"Save folder: {saveFolder}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)

fig = vs.plt.figure(figsize=(10,8))
gs = GridSpec(2, 2, wspace=0, hspace=0.3)

# 图a,b
t = 0.5
dt = 0.02
s = np.linspace(0,1,200)

e = 0.05
for (i,w) in enumerate([0.1,0.3]):
    x = np.random.normal(0,e,s.shape)
    g1 = np.exp(-(s-t)**2/w**2) + x
    g2 = np.exp(-(s-t-dt)**2/w**2) + x
    ax = fig.add_subplot(gs[i])
    ax.plot(s,g1,label="Source 1")
    ax.plot(s,g2,label="Source 2")
    ax.set_title([f"Gaussian, w = 0.1",f"Gaussian, w = 0.3"][i])
    ax.set_ylim([-0.2,1.2])
    ax.set_xlabel("s")
    if not i:
        ax.set_ylabel("g(s)")
    else:
        ax.legend()
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False, left=False)
    ax.text(0.05,0.9,["(a)","(b)"][i],transform=ax.transAxes)




# 图 c, d
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
    x = rS*np.linspace(-1,1,num)
    z = np.sqrt(rS**2 - x**2)
    # zv = np.sqrt(rS**2-xv**2-yv**2)
    B = np.zeros(x.shape)

    for i in range(num):
        r = np.array([x[i],0,z[i]])
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
        B[i] = np.vdot(nSensor,Bi)

    return x,B



axLabels = ["Unshielded, x","Shielded, radial"]
thetas = np.array([30,33])/180*np.pi
p = np.array([0,100e-9,0])
e = 0.2e-12

Bmax = 0
Bmin = np.inf

logging.info("Calculating...")
resultls = []
for (i,s) in enumerate(solvers[::-1]):
    ax = fig.add_subplot(gs[1,i])
    for (j,theta) in enumerate(thetas):
        rp = np.array([np.sin(theta),0,np.cos(theta)])*8e-2
        x,B = getTheoBm(s,rp,p,num=200)

        B += np.random.normal(0,e,x.shape)
        Bmax = max(Bmax,np.nanmax(B))
        Bmin = min(Bmin,np.nanmin(B))
        
        ax.plot(x*1e2,B*1e12,label=f"source {j+1}")
        ax.set_ylim([-4,4.5])

    ax.set_xlabel("x (cm)")
    if not i:
        ax.set_ylabel("B (pT)")
        ax.legend(loc="lower left")
    else:
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelleft=False, left=False)

    ax.text(0.05,0.9,["(c)","(d)"][i],transform=ax.transAxes)
    ax.set_title(axLabels[i])


logging.info(f"B max: {Bmax*1e12:.2f} pT. B min: {Bmin*1e12:.2f} pT.")

fig.savefig(os.path.join(saveFolder,f"GaussianAndB.jpg"),dpi=600)
vs.plt.close(fig)

logging.info("done.")