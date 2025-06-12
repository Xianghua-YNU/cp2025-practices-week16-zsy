"""
学生模板：铝棒热传导问题
文件：heat_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100  # 初始温度
    for t in range(1, Nt):
        for i in range(1, Nx - 1):
            u[i, t] = u[i, t-1] + D * dt / dx**2 * (u[i+1, t-1] - 2*u[i, t-1] + u[i-1, t-1])
        # 边界条件
        u[0, t] = 0
        u[-1, t] = 0

    return u
    
def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, Nt*dt, Nt)
    X, T = np.meshgrid(x, t, indexing='ij')

    u_analytical = np.zeros((Nx, Nt))
    for n in range(1, n_terms + 1):
        kn = n * np.pi
        term = (200 / (n * np.pi)) * np.sin(kn * X) * np.exp(-kn**2 * D * T)
        u_analytical += term

    return u_analytical

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    dt_values = [0.25, 0.5]  # 时间步长
    stability = []

    for dt_val in dt_values:
        r = D * dt_val / dx**2
        if r <= 0.5:
            stability.append("稳定")
        else:
            stability.append("不稳定")

    return stability

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    u = np.zeros((Nx, Nt))
    x = np.linspace(0, L, Nx)
    u[:, 0] = np.where(x < 0.5, 100, 50)  # 左边初始温度100K, 右边初始温度50K

    for t in range(1, Nt):
        for i in range(1, Nx - 1):
            u[i, t] = u[i, t-1] + D * dt / dx**2 * (u[i+1, t-1] - 2*u[i, t-1] + u[i-1, t-1])
        # 边界条件
        u[0, t] = 0
        u[-1, t] = 0

    return u

def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    h = 0.01  # 冷却系数 (s^-1)
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100  # 初始温度

    for t in range(1, Nt):
        for i in range(1, Nx - 1):
            cooling_term = h * dt * (0 - u[i, t-1])  # 牛顿冷却项
            u[i, t] = u[i, t-1] + D * dt / dx**2 * (u[i+1, t-1] - 2*u[i, t-1] + u[i-1, t-1]) + cooling_term
        # 边界条件
        u[0, t] = 0
        u[-1, t] = 0

    return u

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    
    示例:
        >>> u = np.zeros((100, 200))
        >>> plot_3d_solution(u, 0.01, 0.5, 200, "示例")
    """
    x = np.arange(0, L + dx, dx)
    t = np.arange(0, Nt*dt, dt)
    X, T = np.meshgrid(x, t, indexing='ij')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u, cmap='hot')
    ax.set_xlabel('位置 (m)')
    ax.set_ylabel('时间 (s)')
    ax.set_zlabel('温度 (K)')
    ax.set_title(title)
    fig.colorbar(surf)
    
    # 保存为PNG文件
    plt.savefig(f"{title}.png")
    plt.show()

if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能
    
    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导
    
    注意:
        学生需要先实现各任务函数才能正常运行
    """
    print("=== 基本热传导模拟 ===")
    u_basic = basic_heat_diffusion()
    plot_3d_solution(u_basic, dx, dt, Nt, "基本热传导模拟")
    print("=== 解析解计算 ===")
    u_analytical = analytical_solution()
    plot_3d_solution(u_analytical, dx, dt, Nt, "解析解计算")
    print("=== 数值解稳定性分析 ===")
    stability = stability_analysis()
    print(f"时间步长0.25s: {stability[0]}, 时间步长0.5s: {stability[1]}")
    print("=== 不同初始条件模拟 ===")
    u_diff_initial = different_initial_condition()
    plot_3d_solution(u_diff_initial, dx, dt, Nt, "不同初始条件模拟")
    print("=== 包含冷却效应的热传导 ===")
    u_cooling = heat_diffusion_with_cooling()
    plot_3d_solution(u_cooling, dx, dt, Nt, "包含冷却效应的热传导")
    print("=== 铝棒热传导问题学生实现 ===")
    print("请先实现各任务函数后再运行主程序")
