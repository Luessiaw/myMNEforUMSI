import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
# paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 固定深度
# paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,0])

paras.radiusOfSensorShell = 11e-2
paras.noise = 100e-15

paras.GeoRefPos = origin
paras.GeoFieldAtRef = 5e-5*unit_z
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

rB = paras.radiusOfBrain

varName1 = "z"
xs1 = np.linspace(0,rB,25)
xticks1 = xs1*1e2
xticks1 = [f"{x:.1f}" for x in xticks1]
xlabel1 = "$z$ (cm)"

varName2 = "x"
xs2 = np.linspace(-rB,rB,50)
xticks2 = xs2*1e2
xticks2 = [f"{x:.0f}" for x in xticks2]
xlabel2 = "$x$ (cm)"

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    theta = np.arctan2(x2,x1)
    r = np.sqrt(x2**2+x1**2)
    newParas.dipoleRadiusRange = np.array([r,r])
    newParas.dipoleThetaRange = np.array([theta,theta])
    return newParas

if __name__ == "__main__":

    runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
            paras,varFunc,
            verify=True,
            multiprocess=True,
            numOfChannelsForDim2=0,
            numOfChannelsForDim3=64,
            )
