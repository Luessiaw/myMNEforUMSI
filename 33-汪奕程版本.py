import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

##### 参数设置 #####
rh = 0.10 #头半径（m）
rs = 0.11 #探头半径(m)
m = 10000           # 假设划分m个格点
num_sensors = 128  # 探头数量
q_amplitude = 100e-9  # 偶极子强度100 nA*m
rq = np.array([4.5, 0, 4.5 * np.sqrt(3)]) * 1e-2  # 偶极子位置 单位：m
fs = 4096                # 采样率 (Hz)
T = 1.0                  # 信号长度 (秒)
n = int(fs * T)          # 采样点数
f = 20  # 偶极子频率 20Hz
t = np.linspace(0, T, n, endpoint=False)  #时间序列
rg= 0.09   #格点球半径 单位：m
q_unit = np.array([0, 1, 0])  # 假设源方向为 y 轴
ng = np.array([0,0,1]/np.sqrt(3)) #地磁方向单位向量
ngT = ng.T

def calculate_magnetic_field(r, rq, q):

    mu0 = 4 * np.pi * 1e-7
    
    # 转换为numpy数组
    r = np.array(r)
    rq = np.array(rq)
    q = np.array(q)
    
    # 基本向量计算
    R_vec = r - rq
    r_norm = np.linalg.norm(r)
    R_norm = np.linalg.norm(R_vec)
    
    # 交叉乘积 a = q x rq
    a = np.cross(q, rq)
    
    # 计算标量 F
    F = R_norm * (r_norm * R_norm + r_norm**2 - np.dot(r, rq))
    
    # 计算梯度 grad_F
    term1 = (R_norm**2 / r_norm + np.dot(r, R_vec) / R_norm + 2 * R_norm + 2 * r_norm) * r
    term2 = (R_norm + r_norm + np.dot(r, R_vec) / R_norm) * rq
    grad_F = term1 - term2
    
    # 组合成最终磁场 B
    B = (mu0 / (4 * np.pi * F**2)) * (F * a - np.dot(a, r) * grad_F)
    
    return B

##### 从确定的单点源出发计算实际测得的磁场（含噪声） #####
def signal_noise(r_sensor): 
    # 计算峰值磁场 (Peak B)
    B_peak = calculate_magnetic_field(r_sensor, rq, q_amplitude * q_unit)  #三维向量
    # 得到 B(t) 的三个分量
    B_t = np.outer(np.sin(2 * np.pi * f * t), B_peak)   #B_t矩阵维数为 采样点数*3 
    # 添加白噪声
    noise_std_T = 0.1 * 1e-12  # 单位特斯拉 (T)
    # np.random.normal(均值, 标准差, 形状)
    noise = np.random.normal(0, noise_std_T, B_t.shape)
    # 将噪声叠加到原始信号上
    B_t_noisy = B_t + noise
    return np.dot(B_t_noisy,ng)  #只看地磁场方向的分量， 得到的是向量，维数为采样点数


#####  滤去10Hz-40Hz 之外的部分，并逆FFT回去 #####
def signal_fft(r_sensor):
    # 计算 FFT 结果
    fft_values = np.fft.fft(signal_noise(r_sensor)* 1e12)
    # 定义截止频率 (10Hz - 40Hz)
    low_freq = 10
    high_freq = 40
    # 计算对应的频率轴
    frequencies = np.fft.fftfreq(n, d=1/fs)
    # 创建过滤器掩码 (Mask)
    mask_1 = (np.abs(frequencies) < low_freq) | (np.abs(frequencies) > high_freq)
    fft_values[mask_1] = 0
    # 执行逆傅里叶变换 (IFFT) 回到时域
    # 因为原始信号是实数，变换回来可能会有极小的虚数残留（由于计算精度），取 np.real
    filtered_signal_time = np.real(np.fft.ifft(fft_values))
    return filtered_signal_time      #单位pT

##### 生成探头的位置的array #####
#使用斐波那契螺旋算法在 z > 0 的半球面上生成均匀分布的探头
def generate_hemisphere_sensors(num_sensors=128, rs=0.11):
    # 初始化数组
    r_sensors = np.zeros((num_sensors, 3))
    
    # 黄金分割比相关的常数
    phi = (np.sqrt(5) - 1) / 2  
    
    for i in range(num_sensors):
        # 将 z 的范围限制在 [0.1, 1.0] 之间，避免探头太靠近赤道平面
        # 这样可以形成一个完美的“碗状”覆盖
        z_relative = (i + 0.5) / num_sensors 
        
        # 计算当前高度对应的圆半径
        r_at_z = np.sqrt(1 - z_relative**2)
        
        # 黄金螺旋角度
        theta = 2 * np.pi * i * phi
        
        # 转换为笛卡尔坐标
        r_sensors[i, 0] = rs * r_at_z * np.cos(theta) # x 单位 m
        r_sensors[i, 1] = rs * r_at_z * np.sin(theta) # y
        r_sensors[i, 2] = rs * z_relative * rs / rs   # z 保持正值
        # 修正 z 坐标
        r_sensors[i, 2] = rs * z_relative
    
    return r_sensors
r_sensors = generate_hemisphere_sensors()

##### 画出探头的阵列分布图 #####
# 创建画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制探头位置 (单位转换为 cm 方便观察)
# c: 颜色, s: 点的大小, alpha: 透明度
scat = ax.scatter(r_sensors[:, 0]*100, 
                  r_sensors[:, 1]*100, 
                  r_sensors[:, 2]*100, 
                  c='blue', s=30, label='Sensors', edgecolors='k')

# 绘制一个半透明的参考球壳 (rh 单位转为 cm)
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_sph = rh*100 * np.cos(u) * np.sin(v)
y_sph = rh*100 * np.sin(u) * np.sin(v)
z_sph = rh*100 * np.cos(v)
ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.2, linewidth=0.5)

# 设置坐标轴标签和范围
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title('Sensor Distribution (128 Sensors)')
# 规定视角从上往下看
ax.view_init(elev=90, azim=-90)
# 保持比例一致，防止球变椭圆
limit = 12
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_box_aspect([1, 1, 1])
plt.legend()
plt.savefig('sensor Distribution (128 Sensors).png')

##### 构建Bf向量 #####
Bf = np.zeros((num_sensors, n))
for i in range(num_sensors):
    r_sen = r_sensors[i]
    Bf[i] = signal_fft(r_sen)

Bf0 = np.zeros((num_sensors, ))   #不考虑滤波
for i in range(num_sensors):
    r_sen = r_sensors[i]
    Bf0[i] = np.dot(calculate_magnetic_field(r_sen, rq, q_amplitude * q_unit),ngT)  #往地磁方向投影
# 下面假设源沿y方向
I = np.eye(128) #生成128*128的单位矩阵（假设有128个探头） 这里假设矩阵C就是该单位矩阵，因为这里是白噪声，协方差矩阵就是I
lam = 10**(-4)

# ##### 划分格点（只沿x方向直线分布） #####
# # 源中心位置: 单位：m
# center_x = rq[0]
# y_fixed = rq[1]
# z_fixed = rq[2]

# # 在中心点附近划分格点
# x_coords = np.linspace(center_x - 0.01, center_x + 0.1, m)
# grid_points = np.zeros((m, 3))
# grid_points[:, 0] = x_coords
# grid_points[:, 1] = y_fixed
# grid_points[:, 2] = z_fixed

# ##### 在整个球体范围内，在三个维度上分别生成线性分布 #####
# x_range = np.linspace(-rh, rh, m)
# y_range = np.linspace(-rh, rh, m)
# z_range = np.linspace(-rh, rh, m)

# # 生成三维网格
# # indexing='ij' 确保顺序为 (x, y, z)
# X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')

# # 将其展平并合并为 N*3 的维数
# # X.ravel() 会将 (m, m, m) 的矩阵拉直成长度为 m^3 的一维向量
# grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

# # 计算每个点到原点的距离
# dist = np.linalg.norm(grid_points, axis=1)

# # 只保留距球心距离合适的点
# mask = (dist > 0.07) & (dist < 0.10)
# grid_points = grid_points[mask]
# m = grid_points.shape[0]  #更新新的m

###### 生成格点的位置的array（使用斐波那契螺旋算法在 z > 0 的半球面上生成均匀分布的格点）#####
def generate_sphere_grid(num_grid, rg):
    # 初始化数组
    grid_points = np.zeros((num_grid, 3))
    
    # 黄金分割比相关的常数
    phi = (np.sqrt(5) - 1) / 2  
    
    for i in range(num_grid):
        # 映射到 [-1, 1]
        z_relative = 1 - (2 * i + 1) / num_grid 
        
        # 计算当前高度对应的圆半径 (根据勾股定理)
        # 此时 z_relative 在 [-1, 1] 波动，r_at_z 依然正确映射圆周大小
        r_at_z = np.sqrt(max(0, 1 - z_relative**2)) 
        
        # 黄金螺旋角度 (保持不变)
        theta = 2 * np.pi * i * phi
        
        # 转换为笛卡尔坐标
        grid_points[i, 0] = rg * r_at_z * np.cos(theta) # x
        grid_points[i, 1] = rg * r_at_z * np.sin(theta) # y
        grid_points[i, 2] = rg * z_relative             # z 覆盖负值和正值
    
    return grid_points

grid_points = generate_sphere_grid(m, rg)

##### 画出格点的阵列分布图 #####
# 创建画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制格点位置 (单位转换为 cm 方便观察)
# c: 颜色, s: 点的大小, alpha: 透明度
scat = ax.scatter(grid_points[:, 0]*100, 
                  grid_points[:, 1]*100, 
                  grid_points[:, 2]*100, 
                  c='blue', s=30, label='Grids', edgecolors='k')

# 绘制一个半透明的参考球壳 (cm)
u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
x_sph = rh*100 * np.cos(u) * np.sin(v)
y_sph = rh*100 * np.sin(u) * np.sin(v)
z_sph = rh*100 * np.cos(v)
ax.plot_wireframe(x_sph, y_sph, z_sph, color='gray', alpha=0.2, linewidth=0.5)

# 设置坐标轴标签和范围
ax.set_xlabel('X (cm)')
ax.set_ylabel('Y (cm)')
ax.set_zlabel('Z (cm)')
ax.set_title(f'Grids (num: {m})')
# # 规定视角从上往下看
# ax.view_init(elev=90, azim=-90)
# 保持比例一致，防止球变椭圆
limit = 12
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)
ax.set_box_aspect([1, 1, 1])

plt.legend()
plt.savefig(f'Grid Distribution ( {m} grids).png')
plt.show()

##### 构建 Gs 矩阵 #####
Gs = np.zeros((num_sensors, m))

for j in range(m):
    rq = grid_points[j]
    for i in range(num_sensors):
        r_sen = r_sensors[i]
        
        # 计算该格点在对应探头处产生的磁场
        B_val = calculate_magnetic_field(r_sen, rq, q_unit)
        
        # 投影到地磁方向
        Gs[i, j] = np.dot(ngT, B_val)
GT = Gs.T  #Gs矩阵是128*m维，m为格点数

# W矩阵 （m*128维）
W =  GT @ np.linalg.inv(Gs @ GT + lam*I)
Q_star=W @ Bf    #Q_star 为m*4096的矩阵，表示不同时刻不同格点处估计源的强弱

Q_star0=W @ Bf0    #Q_star 为m*1的矩阵，表示不同格点处估计源的强弱
# 保存grid_points和Q_star矩阵
filename1 = "source_estimation_results_3D_noise.npz"
filename2 = "source_estimation_results_3D.npz"

# 使用压缩保存，可以显著减小文件体积
np.savez_compressed(
    filename1, 
    grid=grid_points, 
    q_star=Q_star
)
print(f"数据已保存至 {filename1}")

np.savez_compressed(
    filename2, 
    grid=grid_points, 
    q_star=Q_star0
)
print(f"数据已保存至 {filename2}")

