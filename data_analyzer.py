import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

class DataAnalyzer:
    """
    单摆实验数据分析器
    用于分析实验数据和理论模型之间的关系
    """
    def __init__(self):
        """初始化数据分析器"""
        self.experimental_data = None
        self.theoretical_data = None
        
    def add_experimental_data(self, data):
        """
        添加实验数据
        参数:
        data: 字典形式的实验数据
        """
        self.experimental_data = data
        
    def add_theoretical_data(self, data):
        """
        添加理论模型数据
        参数:
        data: 字典形式的理论数据
        """
        self.theoretical_data = data
    
    def estimate_gravity_from_periods(self, lengths, periods):
        """
        通过线性回归估算重力加速度
        使用公式: T^2 = (4π^2/g) * L
        
        参数:
        lengths: 摆长列表(m)
        periods: 对应的周期列表(s)
        
        返回:
        g_value: 估算的重力加速度(m/s^2)
        g_error: 估算误差
        fit_params: 拟合参数
        """
        # 计算周期平方
        periods_squared = np.square(periods)
        
        # 定义线性函数 y = mx + b
        def linear_func(x, m, b):
            return m * x + b
        
        # 使用最小二乘法拟合数据
        popt, pcov = curve_fit(linear_func, lengths, periods_squared)
        m, b = popt
        
        # 从斜率计算g (g = 4π²/m)
        g_value = 4 * np.pi**2 / m
        
        # 计算g的标准误差
        g_error = 4 * np.pi**2 / m**2 * np.sqrt(pcov[0][0])
        
        return {
            'g_value': g_value,
            'g_uncertainty': g_error,
            'fit_params': popt,
            'fit_covariance': pcov
        }
    
    def analyze_damping(self, times, amplitudes):
        """
        分析阻尼系数
        使用指数衰减模型: A = A₀e^(-γt)
        
        参数:
        times: 时间列表
        amplitudes: 振幅列表
        
        返回:
        initial_amplitude: 初始振幅
        damping_coefficient: 阻尼系数
        """
        # 指数衰减模型
        def exp_decay(t, a0, gamma):
            return a0 * np.exp(-gamma * t)
        
        # 拟合阻尼衰减
        popt, pcov = curve_fit(exp_decay, times, amplitudes)
        a0, gamma = popt
        
        return {
            'initial_amplitude': a0,
            'damping_coefficient': gamma,
            'damping_uncertainty': np.sqrt(pcov[1][1])
        }
    
    def calculate_error_metrics(self):
        """
        计算实验与理论的误差指标
        
        返回:
        metrics: 误差指标字典
        """
        if self.experimental_data is None or self.theoretical_data is None:
            raise ValueError("需要先添加实验数据和理论数据!")
            
        # 确保数据时间点匹配
        exp_times = self.experimental_data['time']
        exp_positions = self.experimental_data['y_position']  # 使用y方向位置
        
        theo_times = self.theoretical_data['time']
        theo_positions = self.theoretical_data['y_position']
        
        # 如果时间点不同，进行插值
        if len(exp_times) != len(theo_times) or not np.allclose(exp_times, theo_times):
            from scipy.interpolate import interp1d
            # 确保理论时间范围包含实验时间范围
            min_time = max(exp_times[0], theo_times[0])
            max_time = min(exp_times[-1], theo_times[-1])
            
            # 过滤时间范围
            exp_mask = (exp_times >= min_time) & (exp_times <= max_time)
            exp_times = exp_times[exp_mask]
            exp_positions = exp_positions[exp_mask]
            
            # 在实验时间点上插值理论数据
            interp_func = interp1d(theo_times, theo_positions, kind='cubic')
            theo_interp = interp_func(exp_times)
        else:
            theo_interp = theo_positions
        
        # 计算误差指标
        errors = exp_positions - theo_interp
        abs_errors = np.abs(errors)
        squared_errors = errors**2
        
        metrics = {
            'RMSE': np.sqrt(np.mean(squared_errors)),  # 均方根误差
            'MAE': np.mean(abs_errors),  # 平均绝对误差
            'Max_Error': np.max(abs_errors),  # 最大误差
            'Error_StdDev': np.std(errors),  # 误差标准差
            'MAPE': np.mean(abs_errors / np.abs(theo_interp)) * 100  # 平均绝对百分比误差
        }
        
        return metrics
    
    def visualize_comparison(self):
        """
        可视化实验与理论数据对比
        
        返回:
        fig: matplotlib图表对象
        """
        if self.experimental_data is None or self.theoretical_data is None:
            raise ValueError("需要先添加实验数据和理论数据!")
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # 1. 位置-时间图
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.experimental_data['time'], self.experimental_data['y_position'], 
                'r-', linewidth=1, label='实验数据')
        ax1.plot(self.theoretical_data['time'], self.theoretical_data['y_position'], 
                'b--', linewidth=1, label='理论预测')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('位置 (m)')
        ax1.set_title('单摆运动轨迹对比')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 相空间图
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.experimental_data['angle'], self.experimental_data['angular_velocity'], 
                'r-', linewidth=1, label='实验数据')
        ax2.plot(self.theoretical_data['angle'], self.theoretical_data['angular_velocity'], 
                'b--', linewidth=1, label='理论预测')
        ax2.set_xlabel('角度 (rad)')
        ax2.set_ylabel('角速度 (rad/s)')
        ax2.set_title('相空间轨迹')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 误差分析图
        ax3 = fig.add_subplot(gs[1, 1])
        
        # 计算误差
        error_metrics = self.calculate_error_metrics()
        
        # 插值处理，确保时间点匹配
        from scipy.interpolate import interp1d
        min_time = max(self.experimental_data['time'][0], self.theoretical_data['time'][0])
        max_time = min(self.experimental_data['time'][-1], self.theoretical_data['time'][-1])
        
        exp_times = self.experimental_data['time']
        exp_positions = self.experimental_data['y_position']
        
        # 过滤时间范围
        exp_mask = (exp_times >= min_time) & (exp_times <= max_time)
        filtered_exp_times = exp_times[exp_mask]
        filtered_exp_positions = exp_positions[exp_mask]
        
        # 在过滤后的实验时间点上插值理论数据
        interp_func = interp1d(self.theoretical_data['time'], 
                              self.theoretical_data['y_position'], kind='cubic')
        theo_interp = interp_func(filtered_exp_times)
        
        # 计算误差
        errors = filtered_exp_positions - theo_interp
        
        ax3.plot(filtered_exp_times, errors, 'g-', alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('误差 (m)')
        ax3.set_title(f'误差分析 (RMSE: {error_metrics["RMSE"]:.6f} m)')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def export_data_to_csv(self, filename):
        """
        将数据导出到CSV文件
        
        参数:
        filename: 输出文件名
        """
        if self.experimental_data is None or self.theoretical_data is None:
            raise ValueError("需要先添加实验数据和理论数据!")
        
        # 创建数据框
        data = pd.DataFrame({
            'Time_exp': self.experimental_data['time'],
            'Angle_exp': self.experimental_data['angle'],
            'Y_Position_exp': self.experimental_data['y_position'],
            'Angular_Velocity_exp': self.experimental_data['angular_velocity'],
            'Time_theo': self.theoretical_data['time'],
            'Angle_theo': self.theoretical_data['angle'],
            'Y_Position_theo': self.theoretical_data['y_position'],
            'Angular_Velocity_theo': self.theoretical_data['angular_velocity']
        })
        
        # 保存到CSV
        data.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")

    def pendulum_length_vs_period_experiment(self, pendulum_class, lengths, gravity=9.8, mass=0.1, damping=0.1, 
                                           initial_angle=np.pi/12, t_span=(0, 20), t_points=1000):
        """
        进行摆长与周期关系的实验
        
        参数:
        pendulum_class: PendulumSimulation类
        lengths: 摆长列表(m)
        gravity: 重力加速度(m/s^2)
        mass: 摆球质量(kg)
        damping: 阻尼系数
        initial_angle: 初始角度(rad)
        t_span: 模拟时间范围
        t_points: 模拟时间点数
        
        返回:
        results: 包含摆长、周期和重力加速度估算的结果字典
        """
        periods = []
        
        # 对每个摆长进行模拟并计算周期
        for length in lengths:
            # 创建单摆实例
            pendulum = pendulum_class(length=length, mass=mass, gravity=gravity, 
                                    damping=damping, initial_angle=initial_angle)
            
            # 运行模拟
            pendulum.simulate(t_span=t_span, t_points=t_points)
            
            # 计算周期
            _, avg_period = pendulum.calculate_periods()
            periods.append(avg_period)
            
            print(f"摆长 = {length:.3f} m, 周期 = {avg_period:.4f} s")
        
        # 使用摆长和周期估算重力加速度
        gravity_estimate = self.estimate_gravity_from_periods(lengths, periods)
        
        # 准备结果
        results = {
            'lengths': lengths,
            'periods': periods,
            'gravity_estimate': gravity_estimate
        }
        
        # 可视化摆长与周期关系
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 摆长与周期关系
        ax1.plot(lengths, periods, 'bo-')
        ax1.set_xlabel('摆长 (m)')
        ax1.set_ylabel('周期 (s)')
        ax1.set_title('摆长与周期关系')
        ax1.grid(True)
        
        # 2. 摆长与周期平方关系
        periods_squared = np.square(periods)
        ax2.plot(lengths, periods_squared, 'ro-')
        
        # 添加拟合线
        m, b = gravity_estimate['fit_params']
        fit_line = m * np.array(lengths) + b
        ax2.plot(lengths, fit_line, 'g--', 
                label=f'拟合线: y = {m:.4f}x + {b:.4f}')
        
        ax2.set_xlabel('摆长 (m)')
        ax2.set_ylabel('周期平方 (s²)')
        ax2.set_title(f'摆长与周期平方关系 (估算重力加速度: {gravity_estimate["g_value"]:.4f} ± {gravity_estimate["g_uncertainty"]:.4f} m/s²)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        return results, fig 