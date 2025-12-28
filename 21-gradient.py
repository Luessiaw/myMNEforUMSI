# 地磁场梯度的容忍度
import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

# paras.dim
paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 位置固定
paras.dipoleThetaRange = np.array([np.pi/3,np.pi/3])
# paras.dipoleThetaRange = np.array([0,0])
paras.dipolePhiRange = np.array([0,0]) # 位置固定

# paras.sensorType
# paras.numOfChannels
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
theta = np.pi/2 # 主磁场方向沿 x 轴
# theta = 0 # 主磁场方向沿 z 轴
paras.GeoFieldAtRef = 5e-5*(unit_x*np.sin(theta)+unit_z*np.cos(theta))
paras.GeoFieldGradientTensor = np.zeros((3,3))
paras.GeoFieldGradientKnown = False

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None
# paras.labelPostfix

# 主磁场方向沿 z 轴
varName = "gradient"
refreshMode = 1 # 需要重新计算 L,W
xs = np.linspace(0,500,11)/1e7 #单位：T/m
xticks = xs*1e7 # 单位：nT/cm


def varFunc3v(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    return newParas

def varFunc1(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[2,2] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

def varFunc2(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[0,2] = x
    tensor[2,0] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

def varFunc3(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    tensor = np.zeros((3,3))
    tensor[0,0] = x
    newParas.GeoFieldGradientTensor = tensor
    return newParas

saveFolder = os.path.join("figs",f"{varName}","Bx-rpx")
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
varFuncs = []

paras3v.varFunc = varFunc3v
parass.append(paras3v)

paras3s1 = deepcopy(paras3s)
paras3s1.labelPostfix = "-1"
paras3s1.varFunc = varFunc1
parass.append(paras3s1)

paras3s2 = deepcopy(paras3s)
paras3s2.labelPostfix = "-2"
paras3s2.varFunc = varFunc2
parass.append(paras3s2)

paras3s3 = deepcopy(paras3s)
paras3s3.labelPostfix = "-3"
paras3s3.varFunc = varFunc3
parass.append(paras3s3)

threads = []
vcs = []
for paras in parass:
    vc = VarContraller(varName,xs,paras.varFunc,paras,refreshMode)
    vcs.append(vc)
    thread = threading.Thread(target=vc.run,kwargs={"saveFolder":saveFolder})
    thread.start()
    threads.append(thread)
for thread in threads:
    thread.join()

fig = vs.plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)

shapes = ["o","d","s","v"]
colors = ["olivedrab","coral","steelblue","hotpink"]

for i,vc in enumerate(vcs):
    vc.saveRes(saveFolder=saveFolder)
    vc.plotRes(ax,xticks,f"-{shapes[i]}",label=vc.baseParas.getLabel(),c=colors[i],fc=colors[i])

ax.set_xlabel("Field Gradient (nT/cm)",fontsize=20)
ax.set_ylabel("Localize Error",fontsize=20)
ax.set_title("",fontsize=24)
ax.set_xscale("linear")
ax.tick_params(axis='both', labelsize=20)

ax.legend()
fig.savefig(os.path.join(saveFolder,f"{varName}-dim{paras3v.dim}"))

print("Done.")



