import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
tic = time.time()

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 1e-2

paras.dipoleStrength = 10e-9
paras.dipoleRadiusRange = np.array([8e-2,9e-2])
paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,np.pi*2])

paras.radiusOfSensorShell = 11e-2
paras.noise = 100e-15

paras.GeoRefPos = origin
paras.GeoFieldAtRef = 5e-5*unit_x
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 0
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 2
paras.fixDipole = None

channels2 = np.arange(5,31,10)
channels3 = np.arange(32,200,64)

for k in range(channels2.size):
    numOfChannelIndex = k
    print(f"Processing num of channels, index = {k}.")
    # paras.noise = noise*1e-15
    varName = f"regularParameter/numOfChannels-index{k}"
    xs = np.arange(-12,3,1)
    xs = np.array([10**x for x in xs.astype(float)])
    xticks = xs
    def varFunc(x,baseParas:Paras):
        newParas = deepcopy(baseParas)
        newParas.regularPara = x
        return newParas

    saveFolder = os.path.join("figs",f"{varName}")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder,exist_ok=True)

    verifyParas(paras,
        save = True,
        saveFolder = os.path.join(varName,"fwd-verify"),
        numOfChannelsForDim2 = channels2[k],
        numOfChannelsForDim3 = channels3[k],
    )

    paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=channels2[k],
                                                    numOfChannelsForDim3=channels3[k])

    fig2 = runVarControllers(varName,xs,xticks,varFunc,
                    paras2v,paras2s,
                    xlabel="Regularization Paramter"
                    )

    fig3 = runVarControllers(varName,xs,xticks,varFunc,
                    paras3v,paras3s,
                    xlabel="Regularization Paramter"
                    )

    for fig in [fig2,fig3]:
        ax = fig.axes[-1]
        ax.set_xscale("log")

    fig2.savefig(os.path.join(saveFolder,f"regularParameter-numOfChannels-index{k}-dim2"))
    fig3.savefig(os.path.join(saveFolder,f"regularParameter-numOfChannels-index{k}-dim3"))

toc = time.time()
print(f"Time used: {toc-tic:.2f} s.")
print("Done.")

