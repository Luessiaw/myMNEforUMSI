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

rp_theta = 30/180*np.pi
rp = np.array([np.cos(rp_theta),0,np.sin(rp_theta)])*8e-2

# rp1 = np.array([0,0,8e-2])
# rp2 = np.array([0,0,7e-2])
# rps = [rp1,rp2]
p = np.array([0,100e-9,0])
# sources = [(rp1,p),(rp2,p)]

paras.gridSpacing = 0.3e-2
paras.fixDipole = (rp,p)
paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                numOfChannelsForDim3=128)

paras3v.theta = 0

paras3s.theta = 0
paras3s.GeoFieldAtRef = 5e-5*(unit_x*np.sin(paras3s.theta)+unit_z*np.cos(paras3s.theta))


for par in [paras3v,paras3s]:
    vz = Visualizer()
    fig,ax = vz.getAxis(par.dim)
    
    rp,p = par.fixDipole
    vz.setAxis(ax,par.dim)
    vz.showHead(par.radiusOfHead,par.dim,ax)
    vz.showSource(rp,p,ax,par.dim,arrowBottomRadius=0.002,arrowTipRadius=0.005,arrowBottomLength=0.02,arrowTipLength=0.01)
    
    sol = Solver(par)
    xv, yv, theoBm = sol.getTheoBm(rp,p,50)
    zv = np.sqrt(par.radiusOfSensorShell**2-xv**2-yv**2)


    trial = sol.singleTrial()
    
    vz = Visualizer()
    vz.showQ(sol,trial.Q,ax)
    # trial.Q
    Bm = trial.Bm

print("Done.")
