# 噪声及灵敏度
import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

# paras.dim
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,9e-2]) # 位置固定
paras.dipoleThetaRange = np.array([0,np.pi/2])
paras.dipolePhiRange = np.array([0,0]) # 位置固定

# paras.sensorType
# paras.numOfChannels
paras.radiusOfSensorShell = 11e-2
paras.intrisicNoise = 500e-15
paras.externalNoise = 0
paras.considerDeadZone = False
paras.deadZoneType = "best" # best, worst, random
paras.axisAngleError = 0 
paras.considerRegistrate = False
paras.registrateType = "best"
paras.registrateError = 0

paras.GeoRefPos = origin
theta = np.pi/2
# theta = 0 # 主磁场方向沿 z 轴
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

varName1 = "noise"
varName2 = "numOfChannels"
refreshMode = 1 # 需要重新计算 L,W

# x1: noise
x1s = np.linspace(0,200,21)/1e15 #单位：T/m
x1ticks = x1s*1e15 # 单位：nT/cm
# x2: numOfChannels
x2s = np.arange(32,200,16) #单位：T/m
x2ticks = x2s # 单位：nT/cm

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.intrisicNoise = x1
    newParas.numOfChannels = x2
    return newParas

saveFolder = os.path.join("figs",f"{varName1}-{varName2}","z")
print(f"Save folder: {saveFolder}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)

# fwd_saveFolder = os.path.join(saveFolder,"fwd-verify")
# if not os.path.exists(fwd_saveFolder):
#     os.makedirs(fwd_saveFolder,exist_ok=True)
# verifyParas(paras,
#     save = True,
#     saveFolder = fwd_saveFolder,
#     numOfChannelsForDim2 = 30,
#     numOfChannelsForDim3 = 64,
# )

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                   numOfChannelsForDim3=64)

parass = []
parass.append(paras3v)
parass.append(paras3s)

def runParas(paras:Paras):

    def singleTrial(solver:Solver,j:int):
        trial = solver.singleTrial()
        solver.trials.addTrial(j,trial)
    
    errs = np.zeros((x1s.size+1,x2s.size+1))
    errs[0,1:] = x2s
    errs[1:,0] = x1s
    vars = np.zeros((x1s.size+1,x2s.size+1))
    vars[0,1:] = x2s
    vars[1:,0] = x1s

    solver = None
    for k1,k2 in tqdm.tqdm(list(itertools.product(range(x1s.size),range(x2s.size))),desc=f"{paras.getLabel()}"):
        x1 = x1s[k1]
        x2 = x2s[k2]
        newParas = deepcopy(paras)
        newParas.intrisicNoise = x1
        newParas.numOfChannels = x2
        if not k1*k2 or not solver: # 初始化
            solver = Solver(newParas)
        else:
            if refreshMode == 1: # 需要重新计算 L 和 W
                solver = Solver(newParas)
            elif refreshMode == 2: # 需要重新计算 W
                solver.W = solver.getInverseMatrix()
            elif refreshMode == 3: # 不需要重新计算
                pass

        # threads = []
        for j in range(solver.paras.numOfTrials):
            singleTrial(solver,j)
        #     thread = threading.Thread(target=singleTrial,args=(solver,j))
        #     thread.start()
        #     threads.append(thread)
        # for thread in threads:
        #     thread.join()

        solver.trials.getMeanTrial()
        errs[k1+1,k2+1] = solver.trials.meanErr
        vars[k1+1,k2+1] = solver.trials.varErr

    filenameErrs = os.path.join(saveFolder,f"results-{paras.getLabel()}-errs.csv")
    filenameVars = os.path.join(saveFolder,f"results-{paras.getLabel()}-vars.csv")

    np.savetxt(filenameErrs,errs,delimiter=",")
    np.savetxt(filenameVars,vars,delimiter=",")

    fig = vs.plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)
    X2,X1 = np.meshgrid(x2ticks,x1ticks)
    vs.plt.pcolormesh(X2,X1,errs[1:,1:]*1e2,shading="auto",cmap="viridis")
    vs.plt.colorbar()
    vs.plt.xlabel(varName2)
    vs.plt.ylabel(varName1)

    fig.savefig(os.path.join(saveFolder,f"{paras.getLabel()}"))

threads = []
for paras in parass:
    thread = threading.Thread(target=runParas,args=(paras,))
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()

# X2,X1 = np.meshgrid(x2ticks,x1ticks)
# for i,vc in enumerate(vcs):
#     fig = vs.plt.figure(figsize=(10,8))
#     ax = fig.add_subplot(1,1,1)
#     # vc.saveRes(saveFolder=saveFolder)
#     vs.plt.pcolormesh(X2,X1,vc.errs*1e2,shading="auto",cmap="viridis")
#     vs.plt.colorbar()
#     vs.plt.xlabel("Noise Level (fT/rtHz)")
#     vs.plt.ylabel("Number of Channels")
#     # vs.plt.gca().set_aspect("equal")

#     fig.savefig(os.path.join(saveFolder,f"{vc.baseParas.getLabel()}"))

print("Done.")



