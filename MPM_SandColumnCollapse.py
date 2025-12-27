import taichi as ti
ti.init(arch=ti.gpu)

dim=2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4
p_rho = 1
p_vol = (dx * 0.5) ** 2
E = 1000
nu = 0.3
friction_angle = 35
sin_phi = ti.sin(ti.pi * friction_angle / 180)    #计算机中没有角度的概念，只有弧度，所以需要进行一个转换
# 这里用到的是taichi的函数，所以用的是ti.sin，目的是告诉计算机之后在gpu进行编写
alpha = (1.633 * sin_phi) / (3 - sin_phi)
# 运用摩尔库伦定律不好在数值模拟中进行描述，因为其屈服面为六边形，有尖角，导数不连续，所以用圆锥DP来近似，这里的alpha就是将摩擦角转换成DP系数α
flip_ratio = 0.95  # FLIP/PIC混合比例

#--------变量场声明--------
x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # 质点位置
v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # 质点速度
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # 质点形变梯度
Jp = ti.field(dtype=float, shape=n_particles)  # 质点体积变化
g_vector = ti.Vector([0, -9.8])  # 重力加速度的向量表示，之后就可以直接向量相加

# 网格相关变量
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid))  # 网格节点速度
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # 网格节点质量
grid_old_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid))  # 网格节点上一步的速度，用于FLIP计算
# grid_force = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid)),虽然用到的F=ma，但是力并没有显式地计算出来，所以不需要这个变量