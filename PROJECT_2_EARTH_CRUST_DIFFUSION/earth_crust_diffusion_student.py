"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt


tau = 365  # 天
A = 10.0 + 273.15   # 转换为开尔文
B = 12.0            # 温度变化幅度保持不变
D = 0.1    # m²/day⁻¹
T_bottom = 11.0 + 273.15 
T_initial = 10.0 + 273.15                          # 深度20米处的固定温度转换为开尔文
depth_max = 20.0  # 最大模拟深度(m)

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # TODO: 设置物理参数
    # TODO: 初始化数组
    # TODO: 实现显式差分格式
    # TODO: 返回计算结果

    # 稳定性参数检查
    r = h * D / a**2
    print(f"稳定性参数 r = {r:.4f}")

    T = np.zeros((n_z, n_t)) + T_initial

    # 初始温度场(地表以下20米全年温度近似为11°C，转换为开尔文)
    T [-1, 0] = T_bottom

    # 时间循环
    for year in range(years):
        for j in range(1, N-1):
        # 地表温度(时变边界条件)
             T [0, j] = A + B * np.sin(2 * np.pi * j / tau)
        # 应用上边界条件
             T[1:-1, j+1] = T[1:-1, j] + r * (T[2:, j] + T[:-2, j] - 2*T [1:-1, j])
        
    depth_array = np.arange(0, DEPTH_MAX + h, h)
    
    return depth, T

def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365])
     plt.figure(figsize=(10, 8))
    
    # 绘制各季节的温度轮廓
    for i, day in enumerate(seasons):
        plt.plot(depth, temperature[:, day], 
                label=f'Day {day}', linewidth=2)
    plt.xlabel('Depth (m)')
    plt.ylabel('Temperature (K)')
    plt.title('Seasonal Temperature Profiles')
    plt.legend()
    plt.grid(True)
    plt.savefig('seasonal_temperature_profile.png')  # 保存为PNG文件
    plt.show()

if __name__ == "__main__":
        depth, T = solve_earth_crust_diffusion()
        plot_seasonal_profiles(depth, T)
        
        
