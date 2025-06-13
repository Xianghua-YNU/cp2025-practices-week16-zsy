#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        # TODO: 创建零数组
        # TODO: 设置初始条件（10 <= x <= 11 区域为1）
        # TODO: 应用边界条件
        u_0 = np.zeros(self.nx)
        mask = (self.x >= 10) & (self.x <= 11)
        u_0[mask] = 1.0
        u_0[0] = 0.0
        u_0[-1] = 0.0
        return u_0
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 显式差分法直接从当前时刻计算下一时刻的解
        数值方法: 使用scipy.ndimage.laplace计算空间二阶导数
        稳定性条件: r = alpha * dt / dx² <= 0.5
        
        实现步骤:
        1. 检查稳定性条件
        2. 初始化解数组和时间
        3. 时间步进循环
        4. 使用laplace算子计算空间导数
        5. 更新解并应用边界条件
        6. 存储指定时间点的解
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算稳定性参数 r = alpha * dt / dx²
        # TODO: 检查稳定性条件 r <= 0.5
        # TODO: 初始化解数组和时间变量
        # TODO: 创建结果存储字典
        # TODO: 存储初始条件
        # TODO: 时间步进循环
        #   - 使用 laplace(u) 计算空间二阶导数
        #   - 更新解：u += r * laplace(u)
        #   - 应用边界条件
        #   - 在指定时间点存储解
        # TODO: 返回结果字典
        r = self.alpha * dt / (self.dx ** 2)

        # 检查稳定性条件 r <= 0.5
        if r > 0.5:
            print(f"Warning: Stability condition violated! r = {r:.4f} > 0.5")
            print(f"Consider reducing dt to < {0.5 * self.dx**2 / self.alpha:.6f}")

        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        nt = int(self.T_final / dt) + 1
        results = {'times': [], 'solutions': [], 'method': 'Explicit FTCS'}

        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Apply Laplacian using scipy.ndimage.laplace
            du_dt = r * laplace(u)
            u += du_dt
            
            # Apply boundary conditions
            u[0] = 0.0
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 隐式差分法在下一时刻求解线性方程组
        数值方法: 构建三对角矩阵系统并求解
        优势: 无条件稳定，可以使用较大时间步长
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建三对角系数矩阵
        3. 时间步进循环
        4. 构建右端项
        5. 求解线性系统
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算扩散数 r
        # TODO: 构建三对角矩阵（内部节点）
        #   - 上对角线：-r
        #   - 主对角线：1 + 2r
        #   - 下对角线：-r
        # TODO: 初始化解数组和结果存储
        # TODO: 时间步进循环
        #   - 构建右端项（内部节点）
        #   - 使用 scipy.linalg.solve_banded 求解
        #   - 更新解并应用边界条件
        # TODO: 返回结果字典
        r = self.alpha * dt / (self.dx ** 2)
        nt = int(self.T_final / dt) + 1
        u = self.u_initial.copy()
    # 构造三对角矩阵（内部节点）
    # 上对角线: -r
    # 主对角线: 1 + 2r
    # 下对角线: -r
        num_internal = self.nx - 2
        banded_matrix = np.zeros((3, num_internal))
        banded_matrix[0, 1:] = -r  # Upper diagonal
        banded_matrix[1, :] = 1 + 2*r  # Main diagonal
        banded_matrix[2, :-1] = -r  # Lower diagonal
        
        # Storage for results
        results = {'times': [], 'solutions': [], 'method': 'Implicit BTCS'}
        
        # Store initial condition
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Right-hand side (internal nodes only)
            rhs = u[1:-1].copy()
            
            # Solve tridiagonal system
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix, rhs)
            
            # Update solution
            u[1:-1] = u_internal_new
            u[0] = 0.0  # Boundary conditions
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results

   
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: Crank-Nicolson方法结合显式和隐式格式
        数值方法: 时间上二阶精度，无条件稳定
        优势: 高精度且稳定性好
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建左端矩阵 A
        3. 时间步进循环
        4. 构建右端向量
        5. 求解线性系统 A * u^{n+1} = rhs
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 计算扩散数 r
        # TODO: 构建左端矩阵 A（内部节点）
        #   - 上对角线：-r/2
        #   - 主对角线：1 + r
        #   - 下对角线：-r/2
        # TODO: 初始化解数组和结果存储
        # TODO: 时间步进循环
        #   - 构建右端向量：(r/2)*u[:-2] + (1-r)*u[1:-1] + (r/2)*u[2:]
        #   - 求解线性系统
        #   - 更新解并应用边界条件
        # TODO: 返回结果字典
        r = self.alpha * dt / (self.dx ** 2)

        nt = int(self.T_final / dt) + 1
        
        # Initialize
        u = self.u_initial.copy()
        
        # Build coefficient matrices for internal nodes
        num_internal = self.nx - 2
        
        # Left-hand side matrix A
        banded_matrix_A = np.zeros((3, num_internal))
        banded_matrix_A[0, 1:] = -r/2  # Upper diagonal
        banded_matrix_A[1, :] = 1 + r  # Main diagonal
        banded_matrix_A[2, :-1] = -r/2  # Lower diagonal
        
        # Storage for results
        results = {'times': [], 'solutions': [], 'method': 'Crank-Nicolson'}
        
        # Store initial condition
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # Time stepping
        for n in range(1, nt):
            # Right-hand side vector
            u_internal = u[1:-1]
            rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
            
            # Solve tridiagonal system A * u^{n+1} = rhs
            u_internal_new = scipy.linalg.solve_banded((1, 1), banded_matrix_A, rhs)
            
            # Update solution
            u[1:-1] = u_internal_new
            u[0] = 0.0  # Boundary conditions
            u[-1] = 0.0
            
            t = n * dt
            
            # Store solution at specified times
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in [res_t for res_t in results['times']]:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
        return results

    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
            
        物理背景: 将PDE转化为ODE系统
        数值方法: 使用laplace算子计算空间导数
        
        实现步骤:
        1. 重构包含边界条件的完整解
        2. 使用laplace计算二阶导数
        3. 返回内部节点的导数
        """
        # TODO: 重构完整解向量（包含边界条件）
        # TODO: 使用 laplace(u_full) / dx² 计算二阶导数
        # TODO: 返回内部节点的时间导数：alpha * d²u/dx²
        u_full = np.concatenate(([0.0], u_internal, [0.0]))
        d2u_dx2 = laplace(u_full) / (self.dx**2)
    
        return self.alpha * d2u_dx2[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 将PDE转化为ODE系统求解
        数值方法: 使用高精度ODE求解器
        优势: 自适应步长，高精度
        
        实现步骤:
        1. 提取内部节点初始条件
        2. 调用solve_ivp求解ODE系统
        3. 重构包含边界条件的完整解
        4. 返回结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 提取内部节点初始条件
        # TODO: 调用 solve_ivp 求解
        #   - fun: self._heat_equation_ode
        #   - t_span: (0, T_final)
        #   - y0: 内部节点初始条件
        #   - method: 指定的积分方法
        #   - t_eval: plot_times
        # TODO: 重构包含边界条件的完整解
        # TODO: 返回结果字典
        u0_internal = self.u_initial[1:-1]
        
        start_time = time.time()
        
        # Solve ODE system
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times,
            rtol=1e-8,
            atol=1e-10
        )
        
        computation_time = time.time() - start_time
        
        # Reconstruct full solutions with boundary conditions
        results = {
            'times': sol.t.tolist(),
            'solutions': [],
            'method': f'solve_ivp ({method})',
            'computation_time': computation_time
        }
        
        for i in range(len(sol.t)):
            u_full = np.concatenate(([0.0], sol.y[:, i], [0.0]))
            results['solutions'].append(u_full)
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
            
        实现步骤:
        1. 调用所有四种求解方法
        2. 记录计算时间和稳定性参数
        3. 返回比较结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # TODO: 打印求解信息
        # TODO: 调用四种求解方法
        #   - solve_explicit
        #   - solve_implicit
        #   - solve_crank_nicolson
        #   - solve_with_solve_ivp
        # TODO: 打印每种方法的计算时间和稳定性参数
        # TODO: 返回所有结果的字典
        print("Solving heat equation using four different methods...")
        print(f"Domain: [0, {self.L}], Grid points: {self.nx}, Final time: {self.T_final}")
        print(f"Thermal diffusivity: {self.alpha}")
        print("-" * 60)
        
        # Solve with all methods
        methods_results = {}
        
        # Explicit method
        print("1. Explicit finite difference (FTCS)...")
        methods_results['explicit'] = self.solve_explicit(dt_explicit, plot_times)
        print(f"   Computation time: {methods_results['explicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['explicit']['stability_parameter']:.4f}")
        
        # Implicit method
        print("2. Implicit finite difference (BTCS)...")
        methods_results['implicit'] = self.solve_implicit(dt_implicit, plot_times)
        print(f"   Computation time: {methods_results['implicit']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['implicit']['stability_parameter']:.4f}")
        
        # Crank-Nicolson method
        print("3. Crank-Nicolson method...")
        methods_results['crank_nicolson'] = self.solve_crank_nicolson(dt_cn, plot_times)
        print(f"   Computation time: {methods_results['crank_nicolson']['computation_time']:.4f} s")
        print(f"   Stability parameter r: {methods_results['crank_nicolson']['stability_parameter']:.4f}")
        
        # solve_ivp method
        print(f"4. solve_ivp method ({ivp_method})...")
        methods_results['solve_ivp'] = self.solve_with_solve_ivp(ivp_method, plot_times)
        print(f"   Computation time: {methods_results['solve_ivp']['computation_time']:.4f} s")
        
        print("-" * 60)
        print("All methods completed successfully!")
        
        return methods_results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
            
        实现步骤:
        1. 创建2x2子图
        2. 为每种方法绘制不同时间的解
        3. 设置图例、标签和标题
        4. 可选保存图像
        """
        # TODO: 创建 2x2 子图
        # TODO: 为每种方法绘制解曲线
        # TODO: 设置标题、标签、图例
        # TODO: 可选保存图像
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 为每种方法绘制曲线
        for i, (method, result) in enumerate(methods_results.items()):
            ax = axs[i // 2, i % 2]
            times = sorted(result.keys())
            for t in times:
                ax.plot(self.x, result[t], label=f't={t}')
            ax.set_title(method)
            ax.set_xlabel('Position x')
            ax.set_ylabel('T u')
            ax.legend()

        # 设置主标题
        plt.tight_layout()
        plt.suptitle('不同方法求解热传导方程的结果比较', y=1.02)

        # 可选保存图像
        if save_figure:
            plt.savefig(filename)
            print(f"图像已保存到: {filename}")

        # 显示图像
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法
            
        返回:
            dict: 精度分析结果
            
        实现步骤:
        1. 选择参考解
        2. 计算其他方法与参考解的误差
        3. 统计最大误差和平均误差
        4. 返回分析结果
        """
        # TODO: 验证参考方法存在
        # TODO: 计算各方法与参考解的误差
        # TODO: 统计误差指标
        # TODO: 打印精度分析结果
        # TODO: 返回精度分析字典
        ref_result = methods_results[reference_method]

        # 计算其他方法与参考解的误差
        if reference_method not in methods_results:
            raise ValueError(f"Reference method '{reference_method}' not found in results")
        
        reference = methods_results[reference_method]
        accuracy_results = {}
        
        print(f"\nAccuracy Analysis (Reference: {reference['method']})")
        print("-" * 50)
        
        for method_name, results in methods_results.items():
            if method_name == reference_method:
                continue
                
            errors = []
            for i, (ref_sol, test_sol) in enumerate(zip(reference['solutions'], results['solutions'])):
                if i < len(results['solutions']):
                    error = np.linalg.norm(ref_sol - test_sol, ord=2)
                    errors.append(error)
            
            max_error = max(errors) if errors else 0
            avg_error = np.mean(errors) if errors else 0
            
            accuracy_results[method_name] = {
                'max_error': max_error,
                'avg_error': avg_error,
                'errors': errors
            }
            
            print(f"{results['method']:25} - Max Error: {max_error:.2e}, Avg Error: {avg_error:.2e}")
        
        return accuracy_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # TODO: 创建求解器实例
    # TODO: 比较所有方法
    # TODO: 绘制比较图
    # TODO: 分析精度
    # TODO: 返回结果
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=25.0)

    # 比较所有方法
    methods_results = solver.compare_methods(
        dt_explicit=0.01,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF'
    )

    # 绘制比较图并保存
    solver.plot_comparison(
        methods_results,
        save_figure=True,
        filename='heat_equation_comparison.png'
    )

    # 分析精度
    accuracy = solver.analyze_accuracy(methods_results)

    return solver, methods_results, accuracy

if __name__ == "__main__":
    solver, results, accuracy = main()
