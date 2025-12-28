import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,9e-2])
paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,np.pi*2])

paras.radiusOfSensorShell = 11e-2
# paras.noise = 50e-15

paras.GeoRefPos = origin
paras.GeoFieldAtRef = 5e-5*unit_x
paras.GeoFieldGradientTensor = np.zeros((3,3))

# paras.regularPara = 0
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

varName1 = "noise"
xs1 = np.arange(10,210,20)
xticks1 = xs1
xlabel1 = "Noise Level (fT)"

varName2 = "regularPara"
xs2 = np.arange(-12,1,1).astype(float)
xticks2 = xs2
xlabel2 = "Log10 of Regularization Parameter"

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.noise = x1*1e-15
    newParas.regularPara = 10**x2
    return newParas

if __name__ == "__main__":

    runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
            paras,varFunc,
            verify=True,
            multiprocess=True
            )

