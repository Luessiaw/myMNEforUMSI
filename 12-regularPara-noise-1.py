import matplotlib
# matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
tic = time.time()

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 0.8e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,9e-2])
paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,np.pi*2])

paras.radiusOfSensorShell = 11e-2
paras.noise = 50e-15

paras.GeoRefPos = origin
paras.GeoFieldAtRef = 5e-5*unit_x
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 0
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

for noise in [10,50,100,200]:
    print(f"Processing noise = {noise} fT.")
    paras.noise = noise*1e-15
    varName = f"noise{noise}fT"

    xs = np.arange(-12,1,1).astype(float)
    xs = 10**xs
    xticks = xs

    def varFunc(x,baseParas:Paras):
        newParas = deepcopy(baseParas)
        newParas.regularPara = x
        return newParas
    
    saveFolder = os.path.join("figs","regularParameter",f"{varName}")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder,exist_ok=True)

    fwd_saveFolder = os.path.join(saveFolder,"fwd-verify")
    if not os.path.exists(fwd_saveFolder):
        os.makedirs(fwd_saveFolder,exist_ok=True)
    verifyParas(paras,
        save = True,
        saveFolder = fwd_saveFolder,
        numOfChannelsForDim2 = 15,
        numOfChannelsForDim3 = 64,
    )
    paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                                    numOfChannelsForDim3=64)

    fig2 = runVarControllers(varName,xs,xticks,varFunc,
                    paras2v,paras2s,
                    xlabel="Regularization Paramter",
                    xscale="log",
                    saveFolder=saveFolder
                    )

    fig3 = runVarControllers(varName,xs,xticks,varFunc,
                    paras3v,paras3s,
                    xlabel="Regularization Paramter",
                    xscale="log",
                    saveFolder=saveFolder
                    )

toc = time.time()
print(f"Time used: {toc-tic:.2f} s.")
print("Done.")

