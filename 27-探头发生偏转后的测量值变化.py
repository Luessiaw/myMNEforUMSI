# 发生一定偏转后，测量值的变化
# import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
import logging

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

varName = "探头发生偏转"
saveFolder = os.path.join("figs",f"{varName}")
print(f"Save folder: {saveFolder}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)

pv = deepcopy(paras)
pv.sensorType = "vector"
sv = Solver(pv)

ps = deepcopy(paras)
ps.sensorType = "scalar"
ps.GeoRefPos = origin
# theta = np.pi/2
theta = 0 # 主磁场方向沿 z 轴
ps.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
ps.GeoFieldGradientTensor = np.zeros((3,3))
ps.GeoFieldGradientKnown = True
ss = Solver(ps)

def getTheoBm(solver:Solver,rp:np.ndarray,p:np.ndarray,rotAxis=unit_z,rotAng=0,num=100):
    '''rotAxis: 偏转轴，整体偏转。
    rotAng:偏转角度，弧度。'''
    rS = solver.paras.radiusOfSensorShell
    thetas = np.linspace(0,np.pi/2,num)
    phis = np.linspace(0,2*np.pi,num)
    thetav,phiv = np.meshgrid(thetas,phis, indexing='ij')
    xv = rS*np.sin(thetav)*np.cos(phiv)
    yv = rS*np.sin(thetav)*np.sin(phiv)
    zv = rS*np.cos(thetav)
    # points = np.stack((xv, yv, zv), axis=-1)
    # points = points.reshape(-1, 3).transpose()
    # mask = np.einsum("ij,ij->j",points,points) <= rS**2
    # points = points[:,mask]
    # xs = points[0,:]
    # ys = points[1,:]
    # zs = points[2,:]
    B = np.zeros((num,num))

    rotMat = rotationMatrixAboutNByTheta(rotAxis,rotAng)
    for i in range(num):
        for j in range(num):
            r = np.array([xv[i,j],yv[i,j],zv[i,j]])
            r = np.einsum("ij,j->i",rotMat,r)
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

    # 计算边界
    dtheta = thetas[1] - thetas[0]
    theta_edges = np.concatenate([thetas - dtheta/2, [thetas[-1]+dtheta/2]])

    dphi = phis[1] - phis[0]
    phi_edges = np.concatenate([phis - dphi/2, [phis[-1]+dphi/2]])

    # 对应笛卡尔坐标
    x_edges = rS * np.sin(theta_edges[:, None]) * np.cos(phi_edges[None, :])
    y_edges = rS * np.sin(theta_edges[:, None]) * np.sin(phi_edges[None, :])
    return x_edges,y_edges,B

def showMeasurement(xv:np.ndarray,yv:np.ndarray,Bm:np.ndarray,ax:vs.plt.Axes,cb:False):
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

# 第一个点
rp_thetas = np.array([0,30,60,90])/180*np.pi
rotAngs = [0,2,4,6]
nrows,ncols = map(len,[rp_thetas,rotAngs])
BmMaxs = []
BmMins = []
dBMaxs = []
dBMins = []
res = {"vector":[],"scalar":[]}
logging.info("Calculating...")
for s in [sv,ss]:
    logging.info(f"{s.paras.sensorType}")
    for i,rp_theta in enumerate(rp_thetas):
        resI = []
        rp = np.array([8*np.sin(rp_theta),0,8*np.cos(rp_theta)])/1e2
        p = np.array([0,100e-9,0])

        Bm0 = 0
        for j,rotAng in enumerate(rotAngs):
            x_edges,y_edges,Bm = getTheoBm(s,rp,p,unit_z,rotAng/180*np.pi)
            resI.append([x_edges,y_edges,Bm])
            
            if not j:
                Bm0 = Bm
            dB = Bm-Bm0

            BmMaxs.append(np.max(Bm))
            BmMins.append(np.min(Bm))
            dBMaxs.append(np.max(dB))
            dBMins.append(np.min(dB))
        
        res[s.paras.sensorType].append(resI)

BmMax = np.max(BmMaxs)
BmMin = np.min(BmMins)
dBMax = np.max(dBMaxs)
dBMin = np.min(dBMins)

logging.info("calculations done.")

for st,Bms in res.items():
    logging.info(f"Drawing: {st}")
    fig1 = vs.plt.figure(figsize=(15,12))
    fig2 = vs.plt.figure(figsize=(15,12))
    for i in range(len(rp_thetas)):
        Bm0 = Bms[i][0][2]
        for j in range(len(rotAngs)):
            ax1 = fig1.add_subplot(nrows,ncols,4*i+j+1)
            ax2 = fig2.add_subplot(nrows,ncols,4*i+j+1)
            
            x_edges,y_edges,Bm = Bms[i][j]
            if not j:
                dB = Bm
            else:
                dB = Bm - Bm0
            showMeasurement(x_edges,y_edges,np.abs(Bm),ax1,True)
            showMeasurement(x_edges,y_edges,np.abs(dB),ax2,True)

    title1 = f"{st}-Bm"
    title2 = f"{st}-delta B"
    fig1.suptitle(title1)
    fig2.suptitle(title2)
    fig1.savefig(os.path.join(saveFolder,f"cb-{title1}.jpg"),dpi=600)
    fig2.savefig(os.path.join(saveFolder,f"cb-{title2}.jpg"),dpi=600)
    vs.plt.close(fig1)
    vs.plt.close(fig2)

logging.info("done.")