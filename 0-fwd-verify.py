from myMNE import *
# tic = time.time()

paras = Paras()
paras.noise = 200e-15
paras.dipoleStrength = 100e-9
paras.parallel = False

rp = unit_z * 8e-2
p = unit_y * paras.dipoleStrength
paras.fixDipole = (rp,p)
paras.numOfTrials = 1

verifyParas(paras,
    save = True,
    saveFolder = "fwd-verify",
    numOfChannelsForDim2 = 15,
    numOfChannelsForDim3 = 64,
)

vs.plt.show()
print("Done.")