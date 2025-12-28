import numpy as np
import matplotlib.pyplot as plt
import os


def getErr(filePath):
    x = np.linspace(-0.01,0.01,50)
    z = np.linspace(0,0.01,50)
    data = np.load(filePath)
    err = data["errs"]
    vmin = np.min(err)
    vmax = np.max(err)
    return err,vmin,vmax

oris = ["x","z","xAz"]

errs = []
vmins = []
vmaxs = []
for ori in oris:
    folder = f"figs/z-x/geo-{ori}"

    for t in ["s","z"]:
        filePath = os.path.join(folder,f"z-x-2{t}.npz")
        err,vmin,vmax = getErr(filePath)
        errs.append(err)
        vmins.append(vmin)
        vmaxs.append(vmax)
    
    
