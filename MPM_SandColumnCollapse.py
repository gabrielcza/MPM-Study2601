import taichi as ti
ti.init(arch=ti.gpu)

dim=2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid   
#这里的1是指整个模拟区域的大小，而不是之后沙柱的大小，有更加通用的写法
# --- 更好的写法 ---

# 1. 定义模拟区域的大小 (eg. 一个长2米，高1米的水槽)
#  domain_size = ti.Vector([2.0, 1.0]) 

# 2. 定义分辨率 (比如 X 轴想要 256 个格子)
#  n_grid_x = 256

# 3. 动态计算 dx (网格大小)
# dx = 长度 / 格子数
# dx = domain_size[0] / n_grid_x  

# 4. 计算 Y 轴需要多少个格子 (自动适配)
# n_grid_y = int(domain_size[1] / dx)

# 5. 定义网格场 (此时 shape 就是长方形了)
#  grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid_x, n_grid_y))
dt = 1.5e-4    #这里需要考虑时间不长，CFL条件
p_rho = 1
p_vol = (dx * 0.5) ** 2
E = 1000
nu = 0.3
friction_angle = 35
cohesion = 0      #一开始可以设置为0，认为是干沙，目前还没有考虑湿沙的情况
sin_phi = ti.sin(ti.pi * friction_angle / 180)    #计算机中没有角度的概念，只有弧度，所以需要进行一个转换
# 这里用到的是taichi的函数，所以用的是ti.sin，目的是告诉计算机之后在gpu进行编写
alpha = (1.633 * sin_phi) / (3 - sin_phi)
# 运用摩尔库伦定律不好在数值模拟中进行描述，因为其屈服面为六边形，有尖角，导数不连续，所以用圆锥DP来近似，这里的alpha就是将摩擦角转换成DP系数α
flip_ratio = 0.95  # FLIP/PIC混合比例，后续会用到这个，这个可以随便命名，但是之后认定好就行了，不要随便改动

#--------变量场声明--------
x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # 质点位置
v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # 质点速度
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # 质点形变梯度
Jp = ti.field(dtype=float, shape=n_particles)  # 质点体积变化
g_vector = ti.Vector([0, -9.8])  # 重力加速度的向量表示，之后就可以直接向量相加

#-------- 网格相关变量--------
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid))  # 网格节点速度，这里的shape是一个二维的，因为是2D模拟，如果之后变成3d就需要三个n_grid
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # 网格节点质量
grid_old_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid))  # 网格节点上一步的速度，用于FLIP计算
# grid_force = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid)),虽然用到的F=ma，但是力并没有显式地计算出来，所以不需要这个变量

# -------- 核心算法 --------

@ti.kernel
def substep():
    # [A] 重置网格
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_old_v[i, j] = [0, 0] # 清空旧速度缓存

    # [B] P2G: 粒子 -> 网格 (PIC/FLIP 混合)
    for p in x:
        base = (x[p] * (1.0 / dx)).cast(int)   #计算粒子所在的网格单元的左下角索引,.cast(int)是向下取整的函数
        fx = x[p] * (1.0 / dx) - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]   #计算B样条权重函数，二次B样条函数插值，主要插值了周围3x3个网格节点

        # 1. 计算粒子当前的应力 (Stress)
        # 这里逻辑和之前一样：先算 SVD，处理塑性，再算应力
        U, sig, V = ti.svd(F[p])
        J = 1.0
        # SVD的就是调用ti.svd这个函数，返回U，sig，V三个矩阵，因为粒子的形变有拉伸，旋转和剪切等，但是旋转不会产生力，所以要把这些变化剥离开来，其中sig是一个对角矩阵，包含了拉伸信息
        
        # MC/ DP
        epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])   
        #这里取了log对数是为了之后把乘法关系变成加法关系，方便计算。sig[0,0]和sig[1,1]分别是x和y方向的拉伸比，但不一定是严格意义上的x和y方向，他代表的是两个正交的方向，为第一主方向和第二主方向，粒子变化最剧烈的两个轴
        trace_epsilon = epsilon.sum() + ti.log(Jp[p])
        #在大变形总变形为塑性变形和弹性变形相乘，因为这里取了对数，所以是相加。另外，这里的epsilon表示的是弹性变形，既trace_epsilon表示的是总变形，为弹性变形和塑性变形相乘
        epsilon_hat = epsilon - (trace_epsilon / 2.0)  
        #这里的epsilon_hat表示的是去除体积变化后的形变，主要是剪切形变，偏应变。因为这里是2维，所以除以2，如果是3维就除以3
        epsilon_hat_norm = epsilon_hat.norm() + 1e-6
        #norm是求向量的模长，这里加了一个很小的数，防止除0错误，这里的norm是算出该粒子到圆心的距离
        delta_gamma = epsilon_hat_norm + (trace_epsilon * alpha)
        if delta_gamma > 0:
            scale = 1.0 - delta_gamma / epsilon_hat_norm
            epsilon_hat *= scale
        
        epsilon_new = epsilon_hat + (trace_epsilon / 2.0)
        #更新后的偏应变加上体积应变，得到新的总应变，这里的hat已经通过scale去除了塑性的部分，只剩下弹性的部分
        sig[0, 0] = ti.exp(epsilon_new[0])
        sig[1, 1] = ti.exp(epsilon_new[1])
        
        # 此时 F[p] 还是上一步的 F，我们在 P2G 计算力，在 G2P 更新 F
        # 这里的 stress 计算是为了把内力传给网格
        stress_j2 = 2 * nu / (1 - 2 * nu) * trace_epsilon * ti.Matrix.identity(float, 2) + 2 * epsilon_new[0] * ti.Matrix([[1,0],[0,0]]) + 2 * epsilon_new[1] * ti.Matrix([[0,0],[0,1]])
        stress = (E / (1+nu)) * stress_j2
        stress = (-dt * p_vol * 4 * (1.0/dx**2)) * stress
        # 这里的4 * (1/dx^2)是B样条二阶导数的简化系数，这里的stress其实是力密度乘以体积再乘以时间步长，得到的是冲量，之后传递给网格节点，相当于一个动量的变化量，所以在P2G阶段直接加到网格速度上

        # 2. 散射到网格
        # 这里只有PIC/FLIP的动量传递和力传递，没有质量传递
        for i, j in ti.static(ti.ndrange(3, 3)):  # ti.static表示编译期展开循环，提升性能，这里因为用到的是二次 B样条插值，所以是3x3的范围
            offset = ti.Vector([i, j])  #遍历邻居网格节点
            dpos = (offset.cast(float) - fx) * dx #粒子到网格节点的距离向量
            weight = w[i][0] * w[j][1]   #计算权重
            
            # 动量传递：仅 mass * velocity
            grid_v[base + offset] += weight * (p_rho * v[p]) 
            grid_m[base + offset] += weight * p_rho
            # 力的传递
            grid_v[base + offset] += weight * (stress @ dpos)

    # [C] Grid Operations: 网格更新
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j] # 动量 -> 速度，在这之前的grid_v其实是动量，所以要变成速度需要除以质量
            
            # **关键点**：在施加重力前，保存一下当前的网格速度
            # 这是 FLIP 计算 delta v 所需要的 "旧速度"
            grid_old_v[i, j] = grid_v[i, j]
            
            # 施加重力
            grid_v[i, j][1] -= 9.8 * dt
            
            # 边界条件
            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # [D] G2P: 网格 -> 粒子 (PIC / FLIP 混合)
    for p in x:
        base = (x[p] * (1.0 / dx)).cast(int)
        fx = x[p] * (1.0 / dx) - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        v_pic = ti.Vector.zero(float, dim)
        v_flip = v[p] # FLIP 基础是粒子原有的速度
        
        # 我们需要在 G2P 阶段计算局部速度梯度，用来更新 F
        # 因为我们没有存储 C，所以必须现在算
        grad_v = ti.Matrix.zero(float, dim, dim) 

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx)
            weight = w[i][0] * w[j][1]
            
            g_v_new = grid_v[base + offset]      # 受力后的网格速度
            g_v_old = grid_old_v[base + offset]  # 受力前的网格速度
            
            # 1. PIC 部分：直接插值当前网格速度
            v_pic += weight * g_v_new   #PIC只做速度平均
            
            # 2. FLIP 部分：累加网格的速度变化量 (dv = new - old)
            v_flip += weight * (g_v_new - g_v_old)   #FLIP会加上新的速度
            
            
            # 3. 计算速度梯度 (用于更新形变 F)
            # 这里的 4 * (1/dx) 是 B-Spline 导数项的简化系数
            grad_v += 4 * (1.0 / dx) * weight * g_v_new.outer_product(dpos)

        # 混合 PIC 和 FLIP
        v[p] = (1.0 - flip_ratio) * v_pic + flip_ratio * v_flip
        
        # 更新位置
        x[p] += dt * v[p]
        
        # 更新形变梯度 F
        # F_new = (I + dt * grad_v) * F_old
        F[p] = (ti.Matrix.identity(float, dim) + dt * grad_v) @ F[p]

# -------- 初始化与 GUI --------
@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.4, ti.random() * 0.4 + 0.05]
        v[i] = [0, 0]
        F[i] = ti.Matrix.identity(float, dim)
        Jp[i] = 1.0

init()
gui = ti.GUI("PIC/FLIP MPM Sand", res=512, background_color=0x112F41)

while gui.running:
    for i in range(50):
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0xF2E9E4)
    gui.show()