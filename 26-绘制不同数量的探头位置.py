import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()


paras.dim = 3
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,10e-2]) 
paras.dipoleThetaRange = np.array([0,np.pi/3]) 
paras.dipolePhiRange = np.array([0,0]) 

paras.sensorType = "scalar"
# paras.numOfChannels = 64
paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 100e-15
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
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = True

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None
# paras.labelPostfix

saveFolder = os.path.join("figs","探头位置")
print(f"Save folder: {saveFolder}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)


nums = [16,32,48,64,80,96]
# nums = [16,]
# parass = []
for num in nums:
    print(num)
    parasN = deepcopy(paras)
    parasN.numOfChannels = num
    # parass.append(parasN)
    solv = Solver(parasN)
    rs = solv.sensorPoints
    
    # fig = vs.plt.figure(figsize=(6,4))
    # ax = fig.add_subplot(1,1,1,projection="3d")
    # np.savetxt(os.path.join(saveFolder,f"SensorPoints-{num}.txt"),rs.transpose(),fmt="%.4f")
    # vz = Visualizer()
    # vz.showGeometry(solv,True,False,False,True,False,ax)
    # vs.plt.title("Sensor Position")
    # ax.view_init(elev=90, azim=-90)
    # fig.savefig(os.path.join(saveFolder,f"SensorPoints-{num}-topview.jpg"),dpi=600)
    # # vs.plt.show()
    # vs.plt.close(fig)
    
    
    # 求相邻探头的距离
    dis = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            dis[i,j] = np.linalg.norm(rs[:,i]-rs[:,j])
    np.savetxt(os.path.join(saveFolder,f"探头距离-{num}.txt"),dis*100,fmt="%.1f")
    
print("done")