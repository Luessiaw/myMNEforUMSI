import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,8e-2]) # 固定深度
# paras.dipoleThetaRange = np.array([0,np.pi/3])
# paras.dipolePhiRange = np.array([0,np.pi*2])

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

varName1 = "theta"
xs1 = np.linspace(0,np.pi,30)
xticks1 = xs1*180/np.pi
xticks1 = [f"{x:.0f}" for x in xticks1]
xlabel1 = "$\\theta (^\\circ)$"

varName2 = "phi"
xs2 = np.linspace(0,np.pi*2,30)
xticks2 = xs2*180/np.pi
xticks2 = [f"{x:.0f}" for x in xticks2]
xlabel2 = "$\\phi (^\\circ)$"

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.dipoleThetaRange = np.array([x1,x1])
    newParas.dipolePhiRange = np.array([x2,x2])
    return newParas

if __name__ == "__main__":

    runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
            paras,varFunc,
            verify=True,
            multiprocess=True,
            numOfChannelsForDim2=0
            )


