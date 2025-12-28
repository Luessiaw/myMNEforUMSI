import numpy as np
import numpy.linalg as lin
# from sklearn.cluster import KMeans


delta_tensor = lambda n:np.identity(n) # kronecker delta
delta3 = delta_tensor(3)

unit_x = np.array([1,0,0])
unit_y = np.array([0,1,0])
unit_z = np.array([0,0,1])
unit_vector = [unit_x,unit_y,unit_z]
origin = np.array([0,0,0])

epsilon = np.zeros((3,3,3)) # levi-civita 张量。
for i in range(3):
    for j in range(3):
        for k in range(3):
            e_i = unit_vector[i]
            e_j = unit_vector[j]
            e_k = unit_vector[k]
            epsilon[i,j,k] = np.vdot(np.cross(e_i,e_j),e_k)
del i,j,k

def crossMatrix(v:np.ndarray):
    '''v: 3x1 向量。返回 3x3 矩阵，
    叉乘可以写成矩阵形式，即 v x u = M u 
    给定 v, 返回 M'''
    return np.einsum("k, ijk -> ij", v, epsilon)

def uniformlyRandomUnitVector(dimension=3):
    while True:
        n = np.random.uniform(-1,1,dimension)
        if lin.norm(n):
            # 避免出现零矢量
            break
    n = n / lin.norm(n)
    return n

def rotationMatrixAboutNByTheta(n,theta):
    '''绕 n 转动 theta 角的旋转矩阵
    n: 单位向量
    theta: 角度，单位度。'''
    # n = np.reshape(n,(1,3))
    # theta = np.deg2rad(theta)

    a = np.einsum("ikj , k -> ij",epsilon,n)
    b = np.einsum("i,j->ij",n,n)
    
    rotationMatrix = delta3*np.cos(theta) + a * np.sin(theta) + b*(1-np.cos(theta))
    return rotationMatrix

def rotationMatrixFromNToM(n,m):
    '''把 n 转动到 m 的旋转矩阵'''
    n = n/lin.norm(n)
    m = m/lin.norm(m)
    rotation_axis = np.cross(n,m)
    if lin.norm(rotation_axis) < 1e-10: # 二者平行
        return np.identity(3)
    rotation_axis = rotation_axis/lin.norm(rotation_axis)
    rotation_angle = np.arccos(np.vdot(n,m))
    rotation_matrix = rotationMatrixAboutNByTheta(rotation_axis,rotation_angle)
    return rotation_matrix

def fibonacci_sphere(M=100):
    '''产生单位球面上均匀分布的 M 个点
    返回 (cos phi sin theta, sin phi sin theta, cos theta)'''
    gold_ratio = (np.sqrt(5)+1)/2
    
    points = []
    for i in range(M):
        a = i/gold_ratio
        a = a - int(a)
        phi = 2*np.pi*a
        z = 1 - (2*i+1)/M
        theta = np.arccos(z)
        p = (np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),z)
        p = np.array(p)
        # p = np.reshape(p,(1,3))
        points.append(p)

    return points

def fullRankG(G):
    ''' G 是一个 c 列矩阵，每一列记为 g1,g2,g3
    计算 G^T G 的秩，
    如果 rank(G) = c, 则返回 G
    如果 rank(G) < c, 则
        取 g1, g2 组成 G2, 
        如果 rank(G2) = 2, 则返回 G2
        如果 rank(G2) < 1, 取 c1, c3 组成 G3,
            如果 rank(G3) = 2, 则返回 G3
            如果 rank(G3) < 2, 则返回 c1
    '''
    if lin.matrix_rank(G) == 3:
        return G
    G2 = np.transpose(np.concatenate(([G[:,0]],[G[:,1]])))
    if lin.matrix_rank(G2) == 2:
        return G2
    G3 = np.transpose(np.concatenate(([G[:,0]],[G[:,2]])))
    if lin.matrix_rank(G3)== 2:
        return G3
    return G[:,0]

# def getRadiusRange(paras):
#     '''球壳半径范围，用于 dipole_distribute_category == 1 时'''
#     radius_of_head = paras["radius_of_head"]
#     depth_of_dipoles = paras["depth_range_of_dipoles"]
#     radius_range = radius_of_head - depth_of_dipoles
#     radius_range.sort()
#     return radius_range

def getRadiusRange(radius,depth_range):
    '''球壳半径范围，用于 dipole_distribute_category == 1 时'''
    # radius_of_head = paras["radius_of_head"]
    # depth_of_dipoles = paras["depth_range_of_dipoles"]
    radius_range = radius - depth_range
    radius_range.sort()
    return radius_range
    
def seperateRange(x1,x2,delta):
    '''在区间 (x1,x2) 上以 delta 间隔取点，并在最后加入 x2'''
    a = np.arange(x1,x2,delta)
    a = np.concatenate([a,[x2]])
    return a


def generateRandomPointIn2DCircle(r=1,theta1=0,theta2=np.pi*2):
    '''在指定圆周上均匀随机生成一个二维坐标'''
    theta = np.random.rand()*(theta2-theta1)+theta1
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x,y])

def generateRandomPointIn2DRing(r1=0,r2=1,theta1=0,theta2=np.pi*2):
    '''在指定圆环内均匀随机生成一个二维坐标'''
    r = np.sqrt((r2**2-r1**2)*np.random.rand()+r1**2)
    theta = np.random.rand()*(theta2-theta1)+theta1
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.array([x,y])

def generateRandomPointIn3DSphereSurface(r=1,theta1=0,theta2=np.pi,phi1=0,phi2=np.pi*2):
    '''在指定球面上均匀随机生成一个三维坐标'''
    theta = np.random.rand()*(theta2-theta1)+theta1
    phi = np.random.rand()*(phi2-phi1)+phi1
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.cos(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])


def generateRandomPointIn3DShell(r1=0,r2=1,theta1=0,theta2=np.pi,phi1=0,phi2=np.pi*2):
    '''在指定球壳内均匀随机生成一个三维坐标'''
    r = np.cbrt((r2**3-r1**3)*np.random.rand()+r1**3)
    theta = np.random.rand()*(theta2-theta1)+theta1
    phi = np.random.rand()*(phi2-phi1)+phi1
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.cos(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def generateRandomPoint(r1=0,r2=1,theta1=0,theta2=np.pi,phi1=0,phi2=np.pi*2):
    '''在指定球壳内均匀随机生成一个三维坐标'''
    r = np.cbrt((r2**3-r1**3)*np.random.rand()+r1**3)
    theta = np.random.rand()*(theta2-theta1)+theta1
    phi = np.random.rand()*(phi2-phi1)+phi1
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return np.array([x,y,z])

def getSphericalUnitVector(r):
    '''在r处的三个单位向量'''
    e1 = r/np.linalg.norm(r)
    e2 = np.cross(unit_z,e1)
    if np.linalg.norm(e2) > 1e-5:
        e2 = e2/np.linalg.norm(e2)
        e3 = np.cross(e1,e2)
        e3 = e3/np.linalg.norm(e3)
    else:
        e2 = unit_x
        e3 = unit_y
    return e1,e2,e3

def generateRandomTangentVector(r):
    '''在三维球面上指定位置产生一个随机切向的方向'''
    # 使用随动坐标系，三个基向量为 er,etheta,ephi
    e1,e2,e3 = getSphericalUnitVector(r)
    x,y = generateRandomPointIn2DCircle()
    n = x*e2 + y*e3
    n = n/np.linalg.norm(n)
    return n

def getProjectionOperatorForAmbientField(sensorPoints,ng,gradioTensors):
    '''由给定的梯度张量计算投影算符
    sensorPoints: 各探头的位置
    ng: 地磁场方向
    gradioTensors: 限定的梯度自由度，最多5个。
    '''
    if len(gradioTensors)>5:
        raise Exception("给定梯度的自由度超过5个")
    numOfChannels = len(sensorPoints)
    bs = []
    for T in gradioTensors:
        b = np.zeros(numOfChannels)
        for i,r in enumerate(sensorPoints):
            d = np.einsum("ij,j->i",T,r)
            b[i] = np.einsum("i,i->",ng,d)
        if np.linalg.norm(b)>1e-16:
            bs.append(b/np.linalg.norm(b))

    b0 = np.ones(numOfChannels)
    bs.append(b0)

    A = np.vstack(bs).transpose()
    Q,R = np.linalg.qr(A)
    
    es = []
    for i in range(Q.shape[1]):
        e = Q[:,i]
        for b in bs:
            if np.abs(np.dot(b/np.linalg.norm(b),e)) > 1e-15:
                es.append(e)
                break

    P = np.identity(numOfChannels)
    for e in es:
        Pe = np.identity(numOfChannels) - np.einsum("i,j->ij",e,e)
        P = np.dot(Pe,P)

    return P