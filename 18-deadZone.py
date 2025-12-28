import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 固定深度
paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,0])

paras.radiusOfSensorShell = 11e-2
# paras.noise = 100e-15
paras.intrisicNoise = 10e-15
paras.externalNoise = 0
# paras.considerDeadZone = True
# paras.deadZoneType = "best" # best, worst, random
# paras.axisAngleError = 0 

paras.GeoRefPos = origin
theta = np.pi/2
# theta = 0
paras.GeoFieldAtRef = 5e-5*(unit_x*np.cos(theta)+unit_z*np.sin(theta))
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

varName = "intrisicNoise-z"
refreshMode = 2 # 需要重新计算 W
xs = np.linspace(10,200,20)*1e-15
xticks = xs*1e15
def varFunc(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.intrisicNoise = x
    return newParas

saveFolder = os.path.join("figs","regularized-deadzone",f"{varName}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)
fwd_saveFolder = os.path.join(saveFolder,"fwd-verify")
if not os.path.exists(fwd_saveFolder):
    os.makedirs(fwd_saveFolder,exist_ok=True)
# verifyParas(paras,
#     save = True,
#     saveFolder = fwd_saveFolder,
#     numOfChannelsForDim2 = 30,
#     numOfChannelsForDim3 = 128,
# )

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                   numOfChannelsForDim3=64)

paras3sW = deepcopy(paras3s) # considerDeadZone
paras3sW.considerDeadZone = True
paras3sW.deadZoneType = "worst"

paras3sR = deepcopy(paras3s) # considerDeadZone
paras3sR.considerDeadZone = True
paras3sR.deadZoneType = "random"
paras3sR.axisAngleError = 10/180*np.pi

paras3sB = deepcopy(paras3s) # not consider
paras3sB.considerDeadZone = True
paras3sB.deadZoneType = "best"

vcs = []
parass = [paras3sW,paras3sR,paras3sB,paras3v]

fig = vs.plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)

shapes = ["o","d","s","v"]
colors = ["olivedrab","coral","steelblue","hotpink"]

for i,p in enumerate(parass):
    vc = VarContraller(varName,xs,varFunc,p,refreshMode=refreshMode)
    vc.run(saveFolder=saveFolder)
    vc.saveRes(saveFolder=saveFolder)
    vc.plotRes(ax,xticks,f"-{shapes[i]}",label=p.getLabel(),c=colors[i],fc=colors[i])
    vcs.append(vc)

ax.set_xlabel("Noise (fT/rtHz)",fontsize=20)
ax.set_ylabel("Localize Error",fontsize=20)
ax.set_title("",fontsize=24)
ax.set_xscale("linear")
ax.tick_params(axis='both', labelsize=20)

ax.legend()
fig.savefig(os.path.join(saveFolder,f"{varName}-dim{p.dim}"))

print("Done.")



