import matplotlib
matplotlib.use("Agg") # 避免多线程绘图问题
from myMNE import *
tic = time.time()

paras = Paras()

paras.radiusOfHead = 10e-2
paras.radiusOfBrain = 9e-2
paras.gridSpacing = 1e-2

paras.dipoleStrength = 100e-9
paras.dipoleRadiusRange = np.array([8e-2,9e-2])
paras.dipoleThetaRange = np.array([0,np.pi/3])
paras.dipolePhiRange = np.array([0,np.pi*2])

paras.radiusOfSensorShell = 11e-2
paras.noise = 10e-15

paras.GeoRefPos = origin
paras.GeoFieldAtRef = 5e-5*unit_x
paras.GeoFieldGradientTensor = np.zeros((3,3))

paras.regularPara = 0
paras.threshold = 0.5

paras.numOfTrials = 500
paras.parallel = True
paras.numOfSampleToPlot = 0
paras.fixDipole = None

varName = "geoOritation"
xs = np.linspace(0,np.pi,20)
xticks = xs*180/np.pi

def varFunc(x,baseParas:Paras):
    newParas = deepcopy(baseParas)
    unit = unit_x*np.sin(x) + unit_z*np.cos(x)
    newParas.GeoFieldAtRef = np.linalg.norm(baseParas.GeoFieldAtRef)*unit
    return newParas

saveFolder = os.path.join("figs",f"{varName}")
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder,exist_ok=True)

verifyParas(paras,
    save = True,
    saveFolder = os.path.join(varName,"fwd-verify"),
    numOfChannelsForDim2 = 15,
    numOfChannelsForDim3 = 64,
)

paras2v,paras2s,paras3v,paras3s = paras.childParas(numOfChannelsForDim2=15,
                                                   numOfChannelsForDim3=64)

fig2 = runVarControllers(varName,xs,xticks,varFunc,
                  paras2v,paras2s,
                  xlabel="Angle ($^\\circ$)"
                  )


fig3 = runVarControllers(varName,xs,xticks,varFunc,
                  paras3v,paras3s,
                  xlabel="Angle ($^\\circ$)"
                  )

toc = time.time()
print(f"Time used: {toc-tic:.2f} s.")


fig2.savefig(os.path.join(saveFolder,f"{varName}-dim2"))
fig3.savefig(os.path.join(saveFolder,f"{varName}-dim3"))

print("Done.")

