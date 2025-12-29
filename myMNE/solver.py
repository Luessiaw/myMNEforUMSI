import numpy as np
from .mathTools import *
from . import visualise as vs
from copy import deepcopy
from datetime import datetime
import threading,multiprocessing
import os
import tqdm
import itertools

# 装饰器计时
import time
from functools import wraps

mu0 = 4*np.pi * 1e-7 # 真空磁导率，单位 H/m=T*m/A
k0 = 1e-7 # mu0/4pi, 单位同上

def print_runtime(func):
    """打印函数运行时间的装饰器（仅支持同步函数）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.perf_counter()
        print(f"函数 {func.__name__} 运行耗时: {end_time - start_time:.6f} 秒")
        return result
    return wrapper

class Paras:
    def __init__(self) -> None:
        # 介质
        self.dim = 2
        self.radiusOfHead = 10e-2
        self.radiusOfBrain = 9e-2 # 大脑半径
        self.gridSpacing = 1e-2
        # 源。如果出于调试等目的需要指定某个源，请在 runTrail 的部分修改。
        self.dipoleStrength = 10e-9 # 偶极子强度
        self.dipoleRadiusRange = np.array([0,10e-2]) # 偶极子位置范围之径向范围。
        self.dipoleThetaRange = np.array([0,np.pi]) # 偶极子位置范围之极角范围。
        self.dipolePhiRange = np.array([0,np.pi*2]) # 偶极子位置范围之方位角范围。
        # 探头阵列
        self.sensorType = "scalar" # 探头类型。可选 vector 或 scalar
        # self.gradio = False # if use gradiometer
        # self.gradioBaseline = 3e-2 # base line of gradiometer, dimesion: m
        self.numOfChannels = 10
        self.radiusOfSensorShell = 11e-2
        self.intrisicNoise = 10e-15 # intrisic noise, affected by deadZone
        self.externalNoise = 10e-15 # externalNoise, only affected by ambient field
        self.considerDeadZone = False  # 对于标量磁力仪，是否考虑盲区 if True, deadZone is in worst state.
        self.deadZoneType = "best" # best, worst, random
        self.axisAngleError = 0 # randomly rotated laser ori, affects intrisic noise 
        self.considerRegistrate = False # consider registration error or not
        self.registrateType = "random" # best, bias, random
        self.registrateError = 0 # randomly biased position
        # 磁场环境
        self.GeoRefPos = origin # 参考位置，设置为原点
        self.GeoFieldAtRef = 5e-5*unit_x # 参考点处的磁场，作为参考磁场。
        self.GeoFieldGradientTensor = np.zeros((3,3)) # 地磁场一阶梯度张量。
        self.GeoFieldGradientKnown = False # whether the program knows the gradient
        self.RealGeoFieldAtRef = None # real Bg at ref. used for solver.getBm(...). If None, same as self.GeoFieldAtRef
        # 逆问题求解
        self.regularPara = 0 # 正则化参数
        self.threshold = 0.5 # 源定位阈值
        # 其他设置
        self.numOfTrials = 500 # 重复实验的次数
        self.parallel = False # 是否多线程
        self.numOfSampleToPlot = 0 # 在重复实验过程中绘制一些图片，以检验效果。
        self.fixDipole = None # 仅用于测试。通常的格式为 (rp,p)
        self.labelPostfix = "" # postfix for label
        self.varFunc = None

    def info(self):
        pass

    def verify(self,saveName=""):
        if saveName:
            saveName += f"-{self.getLabel()}"
        vz = Visualizer()
        vz.showOneTrial(self,saveName=saveName)

    def childParas(self,numOfChannelsForDim2=15,numOfChannelsForDim3=64):
        '''由自身产生四种paras, 对应维度和探头类型'''
        paras2v = deepcopy(self)
        paras2v.numOfChannels = numOfChannelsForDim2
        paras2v.dim = 2
        paras2v.sensorType = "vector"

        paras2s = deepcopy(self)
        paras2s.numOfChannels = numOfChannelsForDim2
        paras2s.dim = 2
        paras2s.sensorType = "scalar"

        paras3v = deepcopy(self)
        paras3v.numOfChannels = numOfChannelsForDim3
        paras3v.dim = 3
        paras3v.sensorType = "vector"

        paras3s = deepcopy(self)
        paras3s.numOfChannels = numOfChannelsForDim3
        paras3s.dim = 3
        paras3s.sensorType = "scalar"
        
        return paras2v,paras2s,paras3v,paras3s

    def getLabel(self):
        if self.sensorType == "scalar" and self.considerDeadZone:
            deadLabel = self.deadZoneType[0].upper()
        else:
            deadLabel = ""
        if self.considerDeadZone:
            pass
        return f"{self.dim}{self.sensorType[0]}{deadLabel}{self.labelPostfix}"

class Solver:

    class Trial:
        def __init__(self,dim,rp,p,Bm,Q,locPos,locErr,dispersion):
            self.dim = dim
            self.rp = rp
            self.p = p
            self.Bm = Bm
            self.Q = Q
            self.locPos = locPos
            self.locErr = locErr
            self.dispersion = dispersion

    class Trials():
        def __init__(self,num,numOfChannels):
            self.numOfTrials = num

            self.locErrs = np.zeros(num)
            self.locDisps = np.zeros(num)
            self.Bms = np.zeros((num,numOfChannels))
            self.rps = np.zeros((num,3))
            self.locPs = np.zeros((num,3))

        def addTrial(self,k,trial):
            self.locErrs[k] = trial.locErr
            self.locDisps[k] = trial.dispersion
            self.rps[k,:] = trial.rp
            self.locPs[k,:] = trial.locPos
            self.Bms[k,:] = trial.Bm
        
        def getMeanTrial(self):
            self.meanErr = np.sqrt(np.mean(self.locErrs**2))
            self.varErr = np.sqrt(np.var(self.locErrs,ddof=1))
            self.meanDisp = np.sqrt(np.mean(self.locDisps**2))
            self.varDisp = np.sqrt(np.var(self.locDisps,ddof=1))
            self.meanBm = np.linalg.norm(np.mean(self.Bms,axis=0))



    def __init__(self,
                 paras:Paras):
        self.paras = paras
        if self.paras.sensorType == "scalar" and self.paras.considerDeadZone:
            self.paras.considerDeadZone = True
        else:
            self.paras.considerDeadZone = False

        self.sourcePoints = self.getSourcePoints()
        self.numOfSourcePoints = self.sourcePoints.shape[1]

        self.sensorPoints = self.getSensorPoints()
        # self.sensorPointErrs = self.getSensorPointError()

        self.geoFields = self.getGeoField(self.sensorPoints,
                                         self.paras.GeoRefPos,
                                         self.paras.GeoFieldAtRef,
                                         self.paras.GeoFieldGradientTensor) # 每个探头位的主磁场，包括梯度信息
        self.sensorOris = self.getSensorOri()

        self.L = self.getLeadField()
        self.W = self.getInverseMatrix()

        self.trials = self.Trials(self.paras.numOfTrials,self.paras.numOfChannels)

    def getSourcePoints(self):
        rB = self.paras.radiusOfBrain
        if self.paras.dim == 3:
            num = int(2*rB/self.paras.gridSpacing)
            x = np.linspace(-rB,rB,num)
            y = np.linspace(-rB,rB,num)
            z = np.linspace(-rB,rB,num)
            xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
            grid_points = np.stack((xv, yv, zv), axis=-1)
            grid_points = grid_points.reshape(-1, 3).transpose()
        elif self.paras.dim == 2:
            # 只考虑 y=0 平面上源的分布
            num = int(2*rB/self.paras.gridSpacing)
            x = np.linspace(-rB,rB,num)
            z = np.linspace(-self.paras.radiusOfBrain,self.paras.radiusOfBrain,num)
            xv, zv = np.meshgrid(x, z, indexing='ij')
            grid_points = np.stack((xv, zv), axis=-1)
            grid_points = grid_points.reshape(-1,2).transpose()
            ys = np.zeros(grid_points.shape[1])
            grid_points = np.vstack([grid_points,ys])
            grid_points = grid_points[[0,2,1],:]

        grid_norm = np.einsum("ij,ij->j",grid_points,grid_points)
        grid_points = grid_points[:,grid_norm < rB**2] 
        grid_norm = np.einsum("ij,ij->j",grid_points,grid_points)
        points = grid_points[:,grid_norm > 0] # 要把原点也去掉
        
        return points

    def getSensorPoints(self):
        if self.paras.dim == 3:
            # 三维情形，探头在上半球面内近似均匀地分布。
            fibPoints = fibonacci_sphere(self.paras.numOfChannels*2)
            fibPoints = fibPoints[:self.paras.numOfChannels]
            points = np.vstack(fibPoints).transpose()
        else:
            # 二维情形，探头在 y=0 平面的圆弧上均匀分布
            thetas = np.pi/2 + np.linspace(-np.pi/2,np.pi/2,self.paras.numOfChannels)
            xs = np.cos(thetas)
            ys = np.zeros(xs.shape)
            zs = np.sin(thetas)
            points = np.vstack([xs,ys,zs])

        points *= self.paras.radiusOfSensorShell
        return points

    def getSensorPointError(self):
        '''sensor point error due to co-registration'''
        if not self.paras.considerRegistrate:
            errs = np.zeros((3,self.paras.numOfChannels))
        if self.paras.registrateType == "random":
            err = self.paras.registrateError
            errs = np.random.normal(0,err,(3,self.paras.numOfChannels))
        else:
            errs = np.zeros((3,self.paras.numOfChannels))
        return errs

    def getGeoField(self,sensorPoints:np.ndarray,geoRefPos=origin,geoFieldAtRef=unit_x*5e-5,gradientTensor=np.zeros((3,3))):
        '''根据原点处的参考磁场和一阶梯度张量计算各个探头位置处的地磁场矢量'''
        geoFieldAtRef = geoFieldAtRef.reshape((3,1))
        geoRefPos = geoRefPos.reshape((3,1))
        geoField = geoFieldAtRef + np.dot(gradientTensor,sensorPoints-geoRefPos)
        return geoField

    def getSensorOri(self,sensorPoints:np.ndarray=None,realGeoFields=False):
        '''根据原点处的参考磁场和一阶梯度张量计算各个探头位置处的地磁场矢量
        3xn array'''
        sensorOris = []
        if sensorPoints is None:
            sensorPoints = self.sensorPoints
        if self.paras.sensorType == "scalar":
            if self.paras.GeoFieldGradientKnown or realGeoFields:
                if self.paras.RealGeoFieldAtRef is None:
                    realGeoFieldAtRef = self.paras.GeoFieldAtRef
                else:
                    realGeoFieldAtRef = self.paras.RealGeoFieldAtRef
                a = self.getGeoField(sensorPoints,
                                     self.paras.GeoRefPos,
                                     realGeoFieldAtRef,
                                     self.paras.GeoFieldGradientTensor)
            else:
                Bg = self.paras.GeoFieldAtRef.reshape((3,1))
                ng = Bg/np.linalg.norm(Bg)
                a = np.tile(ng,(1,self.geoFields.shape[1]))
        elif self.paras.sensorType == "vector":
            if sensorPoints is None:
                sensorPoints = self.sensorPoints
            a = sensorPoints
        nA = np.sqrt(np.einsum("ij,ij->j",a,a))
        sensorOris = np.einsum("ij,j->ij",a,1/nA)

        return sensorOris

    def getL3x3(self,r:np.ndarray,r_p:np.ndarray):
        '''r_p:源点。r:场点'''
        R = r - r_p
        nr = np.linalg.norm(r)
        nR = np.linalg.norm(R)
        F = nR*(nr*nR+np.vdot(r,R))
        nablaF = (nR**2/nr+nR)*r + (np.vdot(r,R)/nR+nR+2*nr)*R

        M1 = -crossMatrix(R)
        M2 = np.einsum("ljk, l, k, i-> ij", epsilon, r, r_p, nablaF)
        L = k0*M1/F - k0*M2/F**2
        return L

    def generateRandomDipole(self):
        '''随机生成耦极子。'''
        # 位置
        r1,r2 = self.paras.dipoleRadiusRange
        theta1,theta2 = self.paras.dipoleThetaRange
        phi1,phi2 = self.paras.dipoleThetaRange
        if self.paras.dim == 2:
            phi1 = 0
            phi2 = 0
        dipolePos = generateRandomPoint(r1,r2,theta1,theta2,phi1,phi2)

        # 方向
        if self.paras.dim == 2:
            dipoleOri = unit_y
        else:
            dipoleOri = generateRandomTangentVector(dipolePos)

        rp = dipolePos
        p = self.paras.dipoleStrength * dipoleOri
        return rp,p

    def getCov(self):
        '''get noise covariance matrix according to 
        intrisic and external noise'''
        num = self.paras.numOfChannels
        if self.paras.sensorType == "scalar" and self.paras.considerDeadZone and self.paras.deadZoneType == "worst":
            ng = self.sensorOris
            r = self.sensorPoints
            dot = np.einsum("ij,ij->j",ng,r)**2/np.einsum("ij,ij->j",r,r)
            intNoise = self.paras.intrisicNoise/dot
            extNoise = self.paras.externalNoise
            noiseLevel = intNoise + extNoise
            C = np.diag(noiseLevel**2)
        else:
            noiseLevel = self.paras.intrisicNoise + self.paras.externalNoise
            C = np.eye(num,num)*noiseLevel**2
        return C

    def getBm(self,rp:np.ndarray,p:np.ndarray,noised=True):
        '''由给定的偶极子产生测量值'''
        if self.paras.dim == 2:
            Q = np.array([np.dot(p,unit_y)])
        else:
            e1,e2,e3 = getSphericalUnitVector(rp)
            Q = np.array([np.dot(p,e2),np.dot(p,e3)])
        sensorPoints = self.sensorPoints + self.getSensorPointError()
        sensorOris = self.getSensorOri(sensorPoints,realGeoFields=True)
        L = self.getLeadField(rp.reshape((3,1)),sensorPoints=sensorPoints,sensorOris=sensorOris)
        Bm = np.dot(L,Q)
        # np.set_printoptions(precision=2)
        # print(f"{self.paras.getLabel()} {self.paras.theta:.1f},{sensorOris[:,0]},{L[:3,0]},{Bm[:3]}")

        # Bm = np.zeros(self.paras.numOfChannels)
        # for i in range(self.paras.numOfChannels):
        #     n = self.sensorOris[i]
        #     r = self.sensorPoints[i]
        #     L3x3 = self.getL3x3(r,rp)
        #     Bm[i] = np.einsum("i,ij,j->",n,L3x3,p)

        if noised:
            if self.paras.sensorType == "scalar" and self.paras.considerDeadZone and self.paras.deadZoneType == "worst":
                ng = sensorOris
                r = self.sensorPoints
                dot = np.einsum("ij,ij->j",ng,r)**2/np.einsum("ij,ij->j",r,r)
                intNoise = self.paras.intrisicNoise/dot
                noiseLevel = intNoise + self.paras.externalNoise
                cov = np.diag(noiseLevel**2)
            elif self.paras.sensorType == "scalar" and self.paras.considerDeadZone and self.paras.deadZoneType == "random":
                thetas = np.random.normal(np.pi/2,self.paras.axisAngleError,self.paras.numOfChannels)
                intNoise = self.paras.intrisicNoise/np.sin(thetas)**2
                extNoise = np.random.normal(0,self.paras.externalNoise,self.paras.numOfChannels)
                noiseLevel = intNoise + extNoise
                cov = np.diag(noiseLevel**2)
            else:
                noiseLevel = self.paras.intrisicNoise + self.paras.externalNoise
                cov = np.eye(self.paras.numOfChannels,self.paras.numOfChannels)*noiseLevel**2
            if noiseLevel < 1e-20:
                noise = np.zeros(self.paras.numOfChannels)
            else:
                L = np.linalg.cholesky(cov)
                sample = np.random.normal(0,1,self.paras.numOfChannels)
                noise = sample @ L.T
                # noise = np.random.normal(0,self.paras.noise,(self.paras.numOfChannels,))
            Bm += noise
        # print(self.paras.intrisicNoise*1e15,noise*1e12)
        return Bm

    def getTheoBm(self,rp:np.ndarray,p:np.ndarray,num=100):
        '''不使用向量化的编程，直接使用公式计算各个场点处的磁场值，以便与前者比较。'''
        if self.paras.dim == 2:
            thetas = np.linspace(0,np.pi,num)
            xs = np.cos(thetas)*self.paras.radiusOfSensorShell
            ys = np.zeros(xs.shape)
            zs = np.sin(thetas)*self.paras.radiusOfSensorShell
            B = np.zeros(num)

            for i in range(num):
                r = np.array([xs[i],ys[i],zs[i]])
                R = r - rp
                nr = np.linalg.norm(r)
                nR = np.linalg.norm(R)
                F = nR*(nr*nR+np.vdot(r,R))
                nablaF = (nR**2/nr+nR)*r + (np.vdot(r,R)/nR+nR+2*nr)*R
                q = np.cross(p,rp)

                if self.paras.sensorType == "scalar":
                    Bgeo = self.paras.GeoFieldAtRef + np.dot(self.paras.GeoFieldGradientTensor,r)
                    nSensor = Bgeo/np.linalg.norm(Bgeo)
                elif self.paras.sensorType == "vector":
                    nSensor = r/np.linalg.norm(r)

                Bi = k0*(q/F-np.vdot(q,r)*nablaF/F**2)
                Bi = np.vdot(nSensor,Bi)
                B[i] = Bi

            return xs,B

        else:
            rS = self.paras.radiusOfSensorShell
            thetas = np.linspace(0,np.pi/2,num)
            phis = np.linspace(0,2*np.pi,num)
            thetav,phiv = np.meshgrid(thetas,phis, indexing='ij')
            xv = rS*np.sin(thetav)*np.cos(phiv)
            yv = rS*np.sin(thetav)*np.sin(phiv)
            zv = rS*np.cos(thetav)
            points = np.stack((xv, yv, zv), axis=-1)
            points = points.reshape(-1, 3).transpose()
            mask = np.einsum("ij,ij->j",points,points) <= rS**2
            points = points[:,mask]
            xs = points[0,:]
            ys = points[1,:]
            zs = points[2,:]
            B = np.zeros((num,num))

            for i in range(num):
                for j in range(num):
                    r = np.array([xv[i,j],yv[i,j],zv[i,j]])
                    R = r - rp
                    nr = np.linalg.norm(r)
                    nR = np.linalg.norm(R)
                    F = nR*(nr*nR+np.vdot(r,R))
                    nablaF = (nR**2/nr+nR)*r + (np.vdot(r,R)/nR+nR+2*nr)*R
                    q = np.cross(p,rp)

                    if self.paras.sensorType == "scalar":
                        Bgeo = self.paras.GeoFieldAtRef + np.dot(self.paras.GeoFieldGradientTensor,r)
                        nSensor = Bgeo/np.linalg.norm(Bgeo)
                    elif self.paras.sensorType == "vector":
                        nSensor = r/np.linalg.norm(r)

                    Bi = k0*(q/F-np.vdot(q,r)*nablaF/F**2)
                    Bij = np.vdot(nSensor,Bi)
                    B[i,j] = Bij

            return xv,yv,B

    def leadFunc(self,r:np.ndarray,rp:np.ndarray):
        '''r_p:源点。r:场点'''
        R = r - rp
        nr = np.linalg.norm(r)
        nR = np.linalg.norm(R)
        F = nR*(nr*nR+np.vdot(r,R))
        nablaF = (nR**2/nr+nR)*r + (np.vdot(r,R)/nR+nR+2*nr)*R

        # M1 = -crossMatrix(R)
        M1 = np.einsum("k, ijk -> ij", R, epsilon)
        M2 = np.einsum("ljk, l, k, i-> ij", epsilon, r, rp, nablaF)
        L = k0*M1/F - k0*M2/F**2
        return L

    # @print_runtime
    def getLeadField(self,sourcePoints:np.ndarray=None,sensorPoints:np.ndarray=None,sensorOris:np.ndarray=None):
        if sourcePoints is None:
            sourcePoints = self.sourcePoints
        if sensorPoints is None:
            sensorPoints = self.sensorPoints
        if sensorOris is None:
            sensorOris = self.sensorOris
        
        numOfrp = sourcePoints.shape[1]
        numOfr = sensorPoints.shape[1]

        rp = np.transpose(np.broadcast_to(sourcePoints,(numOfr,3,numOfrp)),(0,2,1))
        r = np.transpose(np.broadcast_to(sensorPoints,(numOfrp,3,numOfr)),(2,0,1))
        n = np.transpose(np.broadcast_to(sensorOris,(numOfrp,3,numOfr)),(2,0,1))
        
        R = r - rp
        nr2 = np.einsum("ijk,ijk->ij",r,r)
        nr = np.sqrt(nr2)
        nR2 = np.einsum("ijk,ijk->ij",R,R)
        nR = np.sqrt(nR2)
        
        rdR = np.einsum("ijk,ijk->ij",r,R)
        
        F = nr*nR2 + nR*rdR
        nablaF = np.einsum("ij,ijk->ijk",nR2/nr+nR,r) + np.einsum("ij,ijk->ijk",rdR/nR+nR+2*nr,R)
        
        M1 = np.einsum("ijk,mnk->ijmn",rp,epsilon)
        M2 = np.einsum("lnk,ijl,ijk,ijm->ijmn",epsilon,r,rp,nablaF)
        
        # L3x3 = k0*M1/F - k0*M2/F**2
        L3x31 = k0*np.einsum("ijmn,ij->ijmn",M1,1/F) 
        L3x32 =  - k0*np.einsum("ijmn,ij->ijmn",M2,1/F**2)
        L3x3 = L3x31 + L3x32
        
        if self.paras.dim == 2:
            L = np.einsum("ijm,ijmn,n->ij",n,L3x3,unit_y)
        else:
            nrp2 = np.einsum("ijk,ijk->ij",rp,rp)
            nrp = np.sqrt(nrp2)
            e1 = np.einsum("ijk,ij->ijk",rp,1/nrp)
            e2 = np.einsum("kmn,m,ijn->ijk",epsilon,unit_z,e1)
            ne2 = np.sqrt(np.einsum("ijk,ijk->ij",e2,e2))
            mask1 = ne2<1e-5 # 处理 rp=unit_z的情况
            mask2 = ne2>=1e-5
            e2[mask1,:] = unit_x
            e2[mask2,:] /= ne2[mask2][:,np.newaxis]
            # e2 = np.einsum("ijk,ij->ijk",e2,1/ne2)
           
            e3 = np.einsum("kmn,ijm,ijn->ijk",epsilon,e1,e2)
            ne3 = np.sqrt(np.einsum("ijk,ijk->ij",e3,e3))
            e3 = np.einsum("ijk,ij->ijk",e3,1/ne3)
            
            L2 = np.einsum("ijm,ijmn,ijn->ij",n,L3x3,e2)
            L3 = np.einsum("ijm,ijmn,ijn->ij",n,L3x3,e3)
            
            L = np.hstack([L2,L3])

        return L

    def getInverseMatrix(self):
        LT = np.transpose(self.L)
        LLT = np.dot(self.L,LT)
        # L3 = LLT + self.paras.regularPara * np.identity(LLT.shape[0])
        L3 = LLT
        if self.paras.intrisicNoise>0:
            L3 += self.paras.regularPara * self.getCov()/self.paras.intrisicNoise**2
        else:
            L3 += self.paras.regularPara * np.eye(L3.shape[0])
        L3i = np.linalg.inv(L3)
        W = np.dot(LT,L3i)
        return W

    def applyInverse(self,Bm):
        Q = np.dot(self.W,Bm)
        return Q

    def evaluateLocRes(self,Q,rp):
        '''评估定位精度及弥散度。目前只针对单个偶极子使用。'''        
        points = self.sourcePoints
        if self.paras.dim == 2:
            powers = Q**2
        elif self.paras.dim == 3:
            powers = Q[:self.numOfSourcePoints]**2
            powers += Q[self.numOfSourcePoints:]**2
        powers /= np.max(powers)
        powers[powers<self.paras.threshold] = 0
        powers = powers/np.sum(powers)
        locPos = np.einsum("ij,j->i",points,powers.reshape(powers.shape[0]))
        # locPos = np.average(points,axis=1,weights=powers)

        locErr = np.linalg.norm(locPos-rp)
        deltaR = points-locPos[:,None]
        deltaR2 = np.einsum("ij,ij->j",deltaR,deltaR)
        dispersion = np.dot(deltaR2,powers)

        return locPos,locErr,dispersion

    def singleTrial(self):
        if self.paras.fixDipole is None:
            rp,p = self.generateRandomDipole()
        else:
            rp,p = self.paras.fixDipole

        Bm = self.getBm(rp,p,True)
        Q = self.applyInverse(Bm)
        locPos,locErr,dispersion = self.evaluateLocRes(Q,rp)
        trial = self.Trial(self.paras.dim,rp,p,Bm,Q,locPos,locErr,dispersion)
        return trial

    def runTrials(self):
        for k in range(self.paras.numOfTrials):
            trial = self.singleTrial()
            self.trials.addTrial(k,trial)
 
        self.trials.getMeanTrial()

    def getFisher(self,sourcePoints:np.ndarray,sourceDipoles:np.ndarray):
        '''求给定点处的 Fisher 信息矩阵
        sourcePoints, sourceDipoles: 3xn矩阵, 表示 n 个格点'''
        numOfrp = sourceDipoles.shape[1]
        
    def saveRes(self,saveFolder=""):
        '''保存一次模拟结果'''
        lines = [self.trials.meanErr,self.trials.meanDisp]

        text = ",".join(map(str,lines))
        filename = os.path.join(saveFolder,f"results-{self.paras.getLabel()}.csv")
        with open(filename,"w",encoding="utf-8") as file:
            file.write(text)
        

class Visualizer:
    ''''''
    def __init__(self):
        pass
        
    def showGeometry(self,solver:Solver=None,showHead=True,showBrain=True,showGrid=False,showSensorPos=True,showSensorOri=True,ax:vs.plt.Axes=None):
        if solver.paras.dim == 2:
            
            if showHead:
                circleHead = vs.Circle((0,0),solver.paras.radiusOfHead*1e2,color="black",fill=False,linestyle="-")
                ax.add_patch(circleHead)
            if showBrain:
                circleBrain = vs.Circle((0,0),solver.paras.radiusOfBrain*1e2,color="black",fill=False,linestyle="dashed")
                ax.add_patch(circleBrain)

            if showGrid:
                ps = solver.sourcePoints*1e2
                ax.scatter(ps[0,:],ps[2,:],edgecolors="gray",facecolors="none",s=50,marker="o")

            if showSensorPos:
                ps = solver.sensorPoints*1e2
                ax.scatter(ps[0,:],ps[2,:],c="green",s=50)

            if showSensorOri:
                ps = solver.sensorPoints*1e2
                ns = solver.sensorOris
                ax.quiver(
                    ps[0,:],ps[2,:],
                    ns[0,:],ns[2,:],
                    color='teal',
                    # width=0.05,
                    # headwidth=5,
                )
        
        elif solver.paras.dim == 3:
            
            if showHead:
                vs.plotSphere(origin,solver.paras.radiusOfHead*1e2,ax=ax,color="lightgray",alpha=0.05)

            if showBrain:
                vs.plotSphere(origin,solver.paras.radiusOfBrain*1e2,ax=ax,color="aquamarine",alpha=0.05)

            if showGrid:
                ps = solver.sourcePoints*1e2
                ax.scatter(ps[0,:],ps[1,:],ps[2,:],color="gray",s=50,marker=".")

            if showSensorPos:
                ps = solver.sensorPoints*1e2
                ax.scatter(ps[0,:],ps[1,:],ps[2,:],c="green",s=50)

            if showSensorOri:
                ps = solver.sensorPoints*1e2
                ns = solver.sensorOris*2
                ax.quiver(
                    ps[0,:],ps[1,:],ps[2,:],
                    ns[0,:],ns[1,:],ns[2,:],
                    color='teal',
                    linewidth=2,
                    arrow_length_ratio=0.4,
                )
            
            return

    def setGeoRange(self,solver:Solver,ax:vs.plt.Axes):
        if solver.paras.dim == 2:
            xrange = solver.paras.radiusOfSensorShell*1e2 + 2
            ytop = solver.paras.radiusOfSensorShell*1e2 + 3
            ybottom = solver.paras.radiusOfHead*1e2 + 1
            ax.set_xbound(-xrange,xrange)
            ax.set_ybound(-ybottom,ytop)
            ax.set_xlabel("$x$ (cm)",fontsize=20)
            ax.set_ylabel("$z$ (cm)",fontsize=20)
            ax.axis("equal")
        else:
            ax.set_xlabel("$x$ (cm)",fontsize=20)
            ax.set_ylabel("$y$ (cm)",fontsize=20)
            ax.set_zlabel("$z$ (cm)",fontsize=20)
            ax.axis("equal")


    def showQ(self,solver:Solver,Q:np.ndarray,ax:vs.plt.Axes,scatterSize=30):
        if solver.paras.dim == 2:
            ps = solver.sourcePoints*1e2
            amplitude = Q**2/np.max(Q**2)
            ax.scatter(
                ps[0,:],ps[2,:],s=scatterSize,c=amplitude,cmap="Reds"
            )

        else:
            ps = solver.sourcePoints*1e2
            
            Q1 = Q[:solver.numOfSourcePoints]**2 + Q[solver.numOfSourcePoints:]**2
            amplitude = Q1/np.max(Q1)
            ax.scatter(
                ps[0,:],ps[1,:],ps[2,:],s=scatterSize,c=amplitude,cmap="Reds"
            )

    def showLocPair(self,trial:Solver.Trial,showLink=True,ax:vs.plt.Axes=None):
        '''画出一次模拟的偶极子位置-定位结果'''
        if trial.dim == 2:
            x1,y1,z1 = trial.rp*1e2
            x2,y2,z2 = trial.locPos*1e2
            
            ax.scatter(x1,z1,marker="x",color="blue",s=50)
            ax.scatter(x2,z2,marker="x",color="red",s=50)
            
            if showLink:
                ax.plot([x1,x2],[z1,z2],"--",c="gray")
        else:
            x1,y1,z1 = trial.rp*1e2
            x2,y2,z2 = trial.locPos*1e2
            
            ax.scatter(x1,y1,z1,marker="x",color="blue",s=50)
            ax.scatter(x2,y2,z2,marker="x",color="red",s=50)
            
            if showLink:
                ax.plot([x1,x2],[y1,y2],[z1,z2],"--",c="gray")

    def showMeasurement(self,solver:Solver,trial:Solver.Trial,ax:vs.plt.Axes):
        if solver.paras.dim == 2:
            xs,Bm = solver.getTheoBm(trial.rp,trial.p,num=200)
            ax.plot(xs*1e2,Bm*1e12,color="black")

            xs = solver.sensorPoints[0,:]
            Bre = np.dot(solver.L,trial.Q) # 重建后计算的磁场
            # if not solver.paras.threshold:
            Bre *= np.mean(np.abs(trial.Bm))/np.mean(np.abs(Bre)) # 进行一个缩放
            ax.scatter(xs*1e2,Bre*1e12,marker="^",color="darkorange")
            ax.scatter(xs*1e2,trial.Bm*1e12,marker="d",color="green")

            ax.set_xlabel("$x$ (cm)",fontsize=20)
            ax.set_ylabel("$B$ (pT)",fontsize=20)
        else:
            xv,yv,Bm = solver.getTheoBm(trial.rp,trial.p,num=50)
            norm = vs.plt.Normalize(vmin=Bm.min(), vmax=Bm.max())  # 归一化到 [0,1]
            cmap = vs.cm.rainbow  # 选择颜色映射（如 'viridis', 'jet', 'plasma'）
            colors = cmap(norm(Bm))  
            ax.plot_surface(xv*1e2,yv*1e2,Bm*1e12,facecolors=colors,edgecolor="none",alpha=0.5)

            xs = solver.sensorPoints[0,:]
            ys = solver.sensorPoints[1,:]
            Bre = np.dot(solver.L,trial.Q) # 重建后计算的磁场
            ax.scatter(xs*1e2,ys*1e2,Bre*1e12,c=Bre*1e12,marker="^",cmap="rainbow",edgecolors="gray")
            ax.scatter(xs*1e2,ys*1e2,trial.Bm*1e12,c=trial.Bm*1e12, marker="d",cmap="rainbow",edgecolors="black")

            ax.set_xlabel("$x$ (cm)",fontsize=20)
            ax.set_ylabel("$y$ (cm)",fontsize=20)
            ax.set_zlabel("$B$ (pT)",fontsize=20)

    def showOneTrial(self,paras:Paras,solver:Solver=None,trial:Solver.Trial=None,saveName=""):
        '''为了检验代码或参数，运行一次求解过程并绘图。'''
        if not solver:
            solver = Solver(paras)
        if not trial:
            trial = solver.singleTrial()
    
        fig = vs.plt.figure(figsize=(14,6))
        # geoFig = vs.plt.figure(figsize=(8,6))
        if paras.dim == 2:
            geoAx = fig.add_subplot(1,2,1)
        else:
            geoAx = fig.add_subplot(1,2,1,projection="3d")
        
        self.showGeometry(solver,
                          showHead=True,
                          showGrid=False,
                          showSensorPos=True,
                          showSensorOri=True,
                          ax=geoAx)
        self.showQ(solver,trial.Q,ax=geoAx)
        self.showLocPair(trial,
                         showLink=True,
                         ax=geoAx)
        self.setGeoRange(solver,geoAx)

        # msFig = vs.plt.figure(figsize=(8,6))
        if paras.dim == 2:
            msAx = fig.add_subplot(1,2,2)
        else:
            msAx = fig.add_subplot(1,2,2,projection="3d")
        self.showMeasurement(solver,trial,msAx)

        if saveName:
            fig.savefig(f"{saveName}.png",dpi=300)

        return fig

    def getAxis(self,dim=3):
        '''获得3d'''
        if dim==2:
            return vs.get2dAx()
        else:
            return vs.get3dAx()

    def showSource(self,rp:np.ndarray,p:np.ndarray,ax:vs.plt.Axes,dim=3, 
        arrowBottomRadius=0.05, arrowTipRadius=0.1, 
        arrowBottomLength=0.5, arrowTipLength=0.1,
        **kwargs):
        '''在源所在位置绘制一个箭头。'''
        n = p/np.linalg.norm(p) # 箭头的方向
        if dim == 2:
            return
        if dim == 3:
            arrowBottom = rp
            arrowTip = rp + n*0.1
            vs.draw_arrow(ax,arrowBottom,arrowTip,arrowBottomRadius,arrowTipRadius,arrowBottomLength,arrowTipLength)
            # vs.plot3dArrow(rp,n,None,ax)

    def showHead(self,headRadius:float,dim=3,ax:vs.plt.Axes=None):
        if dim == 3:
            vs.plotSphere(origin,headRadius,ax=ax,color="oldlace",alpha=0.05)

    def setAxis(self,ax:vs.plt.Axes,dim=3):
        if dim==3:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")

class VarContraller:
    def __init__(self,variableName,variableValues,variableFunc,baseParas:Paras,refreshMode=1):
        self.variableName = variableName
        self.variableFunc = variableFunc
        self.baseParas = deepcopy(baseParas)

        self.xs = variableValues
        self.errs = np.zeros(self.xs.shape)
        self.errVars = np.zeros(self.xs.shape)
        self.disps = np.zeros(self.xs.shape)
        self.dispVars = np.zeros(self.xs.shape)

        # 三种刷新模式：
        # if self.variableName in ["dipoleStrength",
        #                          "dipoleRadiusRange",
        #                          "dipoleThetaRange",
        #                          "dipolePhiRange",
        #                          "noise",
        #                          "numOfTrials",
        #                          "threshold"]:
        #     # 每次更新 x 时不需要重新计算 L 或 W
        #     self.refreshMode = 3
        # elif self.variableName in ["regularPara"]:
        #     # 每次更新 x 时不需要重新计算 L, 但需要重新计算 W
        #     self.refreshMode = 2
        # else:
        #     # 每次更新 x 时都要重新计算 L
        #     self.refreshMode = 1
        self.refreshMode = refreshMode

    # @print_runtime
    def run(self,saveFolder=""):

        def singleVar(k,x,solver=None,saveFolder=""):
            if not k or not solver: # 初始化
                paras = self.variableFunc(x,deepcopy(self.baseParas))
                solver = Solver(paras)
            else:
                if self.refreshMode == 1: # 需要重新计算 L 和 W
                    paras = self.variableFunc(x,deepcopy(self.baseParas))
                    solver = Solver(paras)
                elif self.refreshMode == 2: # 需要重新计算 W
                    solver.paras = self.variableFunc(x,deepcopy(self.baseParas))
                    solver.W = solver.getInverseMatrix()
                elif self.refreshMode == 3: # 不需要重新计算
                    solver.paras = self.variableFunc(x,deepcopy(self.baseParas))

            if not saveFolder:
                saveFolder = os.path.join("figs",self.variableName)
            saveFolder = os.path.join(saveFolder,"samples",f"{k}",f"sample-{solver.paras.getLabel()}")
            if solver.paras.parallel:
                threads = []
                for j in range(solver.paras.numOfTrials):
                    thread = threading.Thread(target=singleTrial,args=(solver,k,j,saveFolder))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
            else:
                for j in range(solver.paras.numOfTrials):                
                    singleTrial(solver,k,j,saveFolder)

            solver.trials.getMeanTrial()

            self.errs[k] = solver.trials.meanErr
            self.errVars[k] = solver.trials.varErr
            self.disps[k] = solver.trials.meanDisp
            self.dispVars[k] = solver.trials.varDisp

            return solver

        def singleTrial(solver:Solver,k:int,j:int,saveFolder:str):
            trial = solver.singleTrial()
            solver.trials.addTrial(j,trial)

            if j<solver.paras.numOfSampleToPlot:
                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder,exist_ok=True)
                saveName = os.path.join(saveFolder,f"img-{j}")
                vz = Visualizer()
                fig = vz.showOneTrial(solver.paras,solver=solver,trial=trial,saveName=saveName)
                vs.plt.close(fig)

        solver = None
        for k,x in tqdm.tqdm(list(enumerate(self.xs)),desc=self.baseParas.getLabel()):
        # for k,x in list(enumerate(self.xs)):
            solver = singleVar(k,x,solver,saveFolder)

    def plotRes(self,ax,xs,linestyle="-",label="",c="olivedrab",fc="blue"):
        errs = self.errs*1e2
        errVars = self.errVars*1e2
        ax.plot(xs,errs,linestyle,c=c,label=label)
        ax.fill_between(xs,errs-errVars,errs+errVars,fc=fc,alpha=0.2)

    def saveRes(self,saveFolder=""):
        lines = []
        xs = ",".join(map(str,self.xs))
        errs = ",".join(map(str,self.errs))
        errVars = ",".join(map(str,self.errVars))
        lines.append(xs)
        lines.append(errs)
        lines.append(errVars)

        text = "\n".join(lines)
        filename = os.path.join(saveFolder,f"results-{self.baseParas.getLabel()}.csv")
        with open(filename,"w",encoding="utf-8") as file:
            file.write(text)


class BiVarContraller: # 双变量 
    def __init__(self,variableName1,variableName2,variableValues1,variableValues2,variableFunc,baseParas:Paras):
        self.variableName1 = variableName1
        self.variableName2 = variableName2
        self.variableFunc = variableFunc
        self.baseParas = deepcopy(baseParas)

        self.xs1 = variableValues1
        self.xs2 = variableValues2
        self.errs = np.zeros((self.xs1.size,self.xs2.size))
        self.errVars = np.zeros(self.errs.shape)
        self.disps = np.zeros(self.errs.shape)
        self.dispVars = np.zeros(self.errs.shape)

        refresMode2 = ["regularPara"]
        refresMode3 = ["dipoleStrength",
                                 "dipoleRadiusRange",
                                 "dipoleThetaRange",
                                 "dipolePhiRange",
                                 "noise",
                                 "numOfTrials",
                                 "threshold"]
        
        # 三种刷新模式：
        if (self.variableName1 in refresMode3) and (self.variableName2 in refresMode3):
            # 每次更新 x 时不需要重新计算 L 或 W
            self.refreshMode = 3
        elif ((self.variableName1 in refresMode2) and (self.variableName2 in refresMode3)):
            # 每次更新 x 时不需要重新计算 L, 但需要重新计算 W
            self.refreshMode = 2
        elif ((self.variableName2 in refresMode2) and (self.variableName1 in refresMode3)):
            self.refreshMode = 2
        else:
            # 每次更新 x 时都要重新计算 L
            self.refreshMode = 1

    # @print_runtime
    def run(self,saveFolder=""):

        def singleVar(k1,k2,x1,x2,solver=None,saveFolder=""):
            if not k1*k2 or not solver: # 初始化
                paras = self.variableFunc(x1,x2,deepcopy(self.baseParas))
                solver = Solver(paras)
            else:
                if self.refreshMode == 1: # 需要重新计算 L 和 W
                    paras = self.variableFunc(x1,x2,deepcopy(self.baseParas))
                    solver = Solver(paras)
                elif self.refreshMode == 2: # 需要重新计算 W
                    solver.paras = self.variableFunc(x1,x2,deepcopy(self.baseParas))
                    solver.W = solver.getInverseMatrix()
                elif self.refreshMode == 3: # 不需要重新计算
                    solver.paras = self.variableFunc(x1,x2,deepcopy(self.baseParas))

            if not saveFolder:
                saveFolder = os.path.join("figs",self.variableName)
            saveFolder = os.path.join(saveFolder,"samples",f"{k1}-{k2}",f"sample-{solver.paras.getLabel()}")
            if solver.paras.parallel:
                threads = []
                for j in range(solver.paras.numOfTrials):
                    thread = threading.Thread(target=singleTrial,args=(solver,j,saveFolder))
                    thread.start()
                    threads.append(thread)
                for thread in threads:
                    thread.join()
            else:
                for j in range(solver.paras.numOfTrials):                
                    singleTrial(solver,j,saveFolder)

            solver.trials.getMeanTrial()

            self.errs[k1,k2] = solver.trials.meanErr
            self.errVars[k1,k2] = solver.trials.varErr
            self.disps[k1,k2] = solver.trials.meanDisp
            self.dispVars[k1,k2] = solver.trials.varDisp

            return solver

        def singleTrial(solver:Solver,j:int,saveFolder:str):
            trial = solver.singleTrial()
            solver.trials.addTrial(j,trial)

            if j<solver.paras.numOfSampleToPlot:
                if not os.path.exists(saveFolder):
                    os.makedirs(saveFolder,exist_ok=True)
                saveName = os.path.join(saveFolder,f"img-{j}")
                vz = Visualizer()
                fig = vz.showOneTrial(solver.paras,solver=solver,trial=trial,saveName=saveName)
                vs.plt.close(fig)

        solver = None
        processIndex = 0
        # processIndex = multiprocessing.current_process().name.split("-")[-1]
        # if processIndex == "MainProcess":
        #     processIndex = 0
        # else:
        #     processIndex = int(processIndex)
        for k1,k2 in tqdm.tqdm(list(itertools.product(range(self.xs1.size),range(self.xs2.size))),desc=f"{self.baseParas.getLabel()}"):
            x1 = self.xs1[k1]
            x2 = self.xs2[k2]
            solver = singleVar(k1,k2,x1,x2,solver,saveFolder)

    def saveRes(self,saveFolder=""):
        
        errs = np.zeros(self.errs.shape+np.array((1,1)))
        errs[0,1:] = self.xs2
        errs[1:,0] = self.xs1
        errs[1:,1:] = self.errs

        vars = np.zeros(self.errs.shape+np.array((1,1)))
        vars[0,1:] = self.xs2
        vars[1:,0] = self.xs1
        vars[1:,1:] = self.errVars
        filenameErrs = os.path.join(saveFolder,f"results-{self.baseParas.getLabel()}-errs.csv")
        filenameVars = os.path.join(saveFolder,f"results-{self.baseParas.getLabel()}-vars.csv")

        np.savetxt(filenameErrs,errs,delimiter=",")
        np.savetxt(filenameVars,vars,delimiter=",")

# @print_runtime
def verifyParas(paras:Paras,save=False,saveFolder="",numOfChannelsForDim2=15,numOfChannelsForDim3=64):
    '''通过画图验证所给参数'''
    if save:
        saveName = os.path.join(saveFolder,"img")
    else:
        saveName = ""

    parass = paras.childParas(numOfChannelsForDim2=numOfChannelsForDim2,
                                numOfChannelsForDim3=numOfChannelsForDim3)

    for paras in parass:
        paras.verify(saveName)
    
# @print_runtime
def runVarControllers(varName,xs,xticks,varFunc,
                      pv:Paras=None,
                      ps:Paras=None,
                      xlabel="",
                      ylabel="Localization Accuracy (cm)",
                      title = "",
                      xscale="linear",
                      saveFolder="",
                      refreshMode = 1
                      ):
    vcv = VarContraller(varName,xs,varFunc,pv,refreshMode=refreshMode)
    vcs = VarContraller(varName,xs,varFunc,ps,refreshMode=refreshMode)
    vcv.run(saveFolder=saveFolder)
    vcs.run(saveFolder=saveFolder)    

    vcv.saveRes(saveFolder=saveFolder)
    vcs.saveRes(saveFolder=saveFolder)

    fig = vs.plt.figure(figsize=(10,8))
    ax = fig.add_subplot(1,1,1)

    vcv.plotRes(ax,xticks,"-o","Scalar",c="olivedrab",fc="olivedrab")
    vcs.plotRes(ax,xticks,"-d","Vector",c="coral",fc="coral")

    ax.set_xlabel(xlabel,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=20)
    ax.set_title(title,fontsize=24)
    ax.set_xscale(xscale)
    ax.tick_params(axis='both', labelsize=20)

    fig.savefig(os.path.join(saveFolder,f"{varName}-dim{pv.dim}"))
    return fig

# @print_runtime
def runBiVarControllers(varName1,xs1,xticks1,
                        varName2,xs2,xticks2,
                        varFunc,paras:Paras=None,
                      xlabel1="",
                      xlabel2="",
                      clabel="Localization Accuracy (cm)",
                      title = "",
                      saveFolder="",
                      refreshMode = 0
                      ):
    vc = BiVarContraller(varName1,varName2,xs1,xs2,varFunc,paras)
    if refreshMode:
        vc.refreshMode = refreshMode
    vc.run(saveFolder=saveFolder)

    fig = vs.plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1,1,1)
    aspect =  xs2.size/xs1.size
    im = ax.imshow(vc.errs*1e2,aspect=aspect,origin="lower")
    # 获取主图的坐标范围
    ax_pos = ax.get_position()
    cbar_height = ax_pos.height  # 使用主图高度
    # 创建与主图等高的colorbar
    cax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, cbar_height])
    cb = vs.plt.colorbar(im, cax=cax)
    cb.set_label(clabel,fontsize=20)
    cb.ax.tick_params(labelsize=20)
    
    ax.set_yticks(range(xs1.size),xticks1)
    ax.set_ylabel(xlabel1,fontsize=20)
    ax.set_xticks(range(xs2.size),xticks2)
    ax.set_xlabel(xlabel2,fontsize=20)
    ax.set_title(title,fontsize=24)
    ax.tick_params(axis='both', labelsize=20)

    npzName = os.path.join(saveFolder,f"{varName1}-{varName2}-{paras.getLabel()}.npz")
    np.savez(npzName,xs1=xs1,xs2=xs2,errs=vc.errs)

    figName = os.path.join(saveFolder,f"{varName1}-{varName2}-{paras.getLabel()}")
    fig.savefig(figName)

    return fig

def runBiVar(varName1,xs1,xticks1,xlabel1,
             varName2,xs2,xticks2,xlabel2,
             paras:Paras,varFunc,
             verify = False,
             multiprocess = True,
             numOfChannelsForDim2=15,
             numOfChannelsForDim3=64
             ):
    saveFolder = os.path.join("figs",f"{varName1}-{varName2}")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder,exist_ok=True)

    fwd_saveFolder = os.path.join(saveFolder,"fwd-verify")
    if verify:
        if not os.path.exists(fwd_saveFolder):
            os.makedirs(fwd_saveFolder,exist_ok=True)

        verifyParas(paras,
            save = True,
            saveFolder = fwd_saveFolder,
            numOfChannelsForDim2 = numOfChannelsForDim2,
            numOfChannelsForDim3 = numOfChannelsForDim3,
        )

    parass = paras.childParas(numOfChannelsForDim2=numOfChannelsForDim2,
                              numOfChannelsForDim3=numOfChannelsForDim3)

    if multiprocess:
        processes = []
        for paras in parass:
            p = multiprocessing.Process(
                target=runBiVarControllers,
                args=(
                    varName1,xs1,xticks1,
                    varName2,xs2,xticks2,
                    varFunc,paras,
                ),
                kwargs={
                    "xlabel1":xlabel1,
                    "xlabel2":xlabel2,
                    "saveFolder":saveFolder
                }
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
    else:
        for paras in parass:
            runBiVarControllers(varName1,xs1,xticks1,
                                varName2,xs2,xticks2,
                                varFunc,paras,
                                xlabel1=xlabel1,
                                xlabel2=xlabel2,
                                saveFolder=saveFolder
                                )


