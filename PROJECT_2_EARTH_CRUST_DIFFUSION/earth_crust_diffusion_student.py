"""
学生模板：地壳热扩散数值模拟
文件：earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt

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
    tau = 365  # 天
    A = 10.0 + 273.15   # 转换为开尔文
    B = 12.0            # 温度变化幅度保持不变
    D = 0.1    # m²/day⁻¹
    T_bottom = 11.0 + 273.15  # 深度20米处的固定温度转换为开尔文
    depth_max = 20.0  # 最大模拟深度(m)

    # 网格参数
    n_z = 200  # 深度方向网格点数
    n_t = 365 * 10  # 时间步数(10年)
    dz = depth_max / (n_z - 1)  # 深度步长(m)
    dt = tau / (n_t - 1)  # 时间步长(天)

    # 稳定性参数检查
    r = D * dt / dz**2
    if r > 0.5:
        raise ValueError("稳定性参数r超过0.5，模拟可能不稳定")

    # 初始化深度数组和温度矩阵
    depth_array = np.linspace(0, depth_max, n_z)
    temperature_matrix = np.zeros((n_z, n_t))

    # 初始温度场(地表以下20米全年温度近似为11°C，转换为开尔文)
    temperature_matrix[:, 0] = T_bottom

    # 时间循环
    for t in range(1, n_t):
        time = t * dt  # 当前时间(天)
        # 地表温度(时变边界条件)
        T_surface = A + B * np.sin(2 * np.pi * time / tau)
        # 应用上边界条件
        temperature_matrix[0, t] = T_surface
        # 应用下边界条件(固定温度)
        temperature_matrix[-1, t] = T_bottom
        # 显式差分格式
        for i in range(1, n_z - 1):
            temperature_matrix[i, t] = temperature_matrix[i, t-1] + r * (temperature_matrix[i+1, t-1] - 2 * temperature_matrix[i, t-1] + temperature_matrix[i-1, t-1])

    return depth_array, temperature_matrix

if __name__ == "__main__":
    # 测试代码
    try:
        depth, T = solve_earth_crust_diffusion()
        print(f"计算完成，温度场形状: {T.shape}")
        plt.figure(figsize=(10, 6))
        seasons = ['春分', '夏至', '秋分', '冬至']  # 四季
        time_points = [int(365 * 9 + 0), int(365 * 9 + 90), int(365 * 9 + 180), int(365 * 9 + 270)]  # 对应四季的时间点
        for i, time_point in enumerate(time_points):
            plt.plot(T[:, time_point], depth, label=seasons[i])
        
        plt.xlabel('Temperature (K)')  
        plt.ylabel('Depth (m)')
        plt.title('The crustal temperature varies with depth (第10年四季)')
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()  # 深度轴向下增加
        plt.savefig('seasonal_temperature_profile.png')  # 保存为PNG文件
        plt.show()
    except NotImplementedError as e:
        print(e)
