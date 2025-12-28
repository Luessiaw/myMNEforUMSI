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

paras.regularPara = 1e-4
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

varName1 = "noise"
xs1 = np.arange(10,210,10)
xticks1 = xs1
xlabel1 = "Noise Level (fT)"

def varFunc(x1,x2,baseParas:Paras):
    newParas = deepcopy(baseParas)
    newParas.noise = x1*1e-15
    newParas.numOfChannels = x2
    return newParas

if __name__ == "__main__":

    varName2 = "numOfChannels"
    xs2 = np.arange(10,31,1)
    xticks2 = xs2
    xlabel2 = "Channel Number"

    runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
            paras,varFunc,
            verify=True,
            multiprocess=True,
            numOfChannelsForDim2=15,
            numOfChannelsForDim3=0,
            )

    xs2 = np.arange(16,100,8)
    xticks2 = xs2
    runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
            paras,varFunc,
            verify=True,
            multiprocess=True,
            numOfChannelsForDim2=0,
            numOfChannelsForDim3=64,
            )

