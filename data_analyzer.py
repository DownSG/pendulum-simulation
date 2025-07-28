import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DataAnalyzer:
    """
    Pendulum experiment data analyzer
    Used for analyzing relationship between experimental data and theoretical models
    """
    def __init__(self):
        """Initialize data analyzer"""
        self.experimental_data = None
        self.theoretical_data = None
        
    def add_experimental_data(self, data):
        """
        Add experimental data
        Parameters:
        data: experimental data in dictionary form
        """
        self.experimental_data = data
        
    def add_theoretical_data(self, data):
        """
        Add theoretical model data
        Parameters:
        data: theoretical data in dictionary form
        """
        self.theoretical_data = data
    
    def calculate_error_metrics(self):
        """
        Calculate error metrics between experimental and theoretical data
        
        Returns:
        metrics: dictionary containing various error metrics
        """
        if self.experimental_data is None or self.theoretical_data is None:
            raise ValueError("Both experimental and theoretical data must be added first!")
            
        exp_periods = self.experimental_data.get('periods', [])
        theo_periods = self.theoretical_data.get('periods', [])
        
        if len(exp_periods) != len(theo_periods):
            raise ValueError("Experimental and theoretical data must have same length!")
            
        errors = np.array(exp_periods) - np.array(theo_periods)
        percent_errors = errors / np.array(theo_periods) * 100
        
        metrics = {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(np.abs(percent_errors)),
            'max_error': np.max(np.abs(errors)),
            'error_std': np.std(errors),
            'percent_errors': percent_errors.tolist(),
            'absolute_errors': errors.tolist()
        }
        
        return metrics
        
    def fit_period_length_relation(self, lengths, periods):
        """
        Fit the period-length relation T = 2π√(L/g)
        Used for gravity calculation from experimental data
        
        Parameters:
        lengths: array of pendulum lengths
        periods: array of measured periods
        
        Returns:
        g: estimated gravitational acceleration
        g_uncertainty: uncertainty of gravity estimate
        """
        # For small angles, T = 2π√(L/g) => T² = (4π²/g)·L
        # So we fit T² = a·L, and g = 4π²/a
        
        # Square the periods
        periods_squared = periods**2
        
        # Linear fit T² = a·L, no intercept
        def linear_func(x, a):
            return a * x
            
        # Fit the model
        popt, pcov = curve_fit(linear_func, lengths, periods_squared)
        a = popt[0]
        a_uncertainty = np.sqrt(pcov[0, 0])
        
        # Calculate g and its uncertainty
        g = 4 * np.pi**2 / a
        g_uncertainty = g * (a_uncertainty / a)  # Error propagation
        
        return g, g_uncertainty
        
    def visualize_comparison(self, show=True, save_path=None):
        """
        Visualize comparison between experimental and theoretical data
        
        Parameters:
        show: whether to show the plot
        save_path: path to save the figure
        
        Returns:
        fig: matplotlib figure object
        """
        if self.experimental_data is None or self.theoretical_data is None:
            raise ValueError("Both experimental and theoretical data must be added first!")
            
        # Extract data
        exp_lengths = self.experimental_data.get('lengths', [])
        exp_periods = self.experimental_data.get('periods', [])
        theo_lengths = self.theoretical_data.get('lengths', [])
        theo_periods = self.theoretical_data.get('periods', [])
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Period vs Length plot
        ax1 = fig.add_subplot(221)
        ax1.plot(exp_lengths, exp_periods, 'o', label='实验数据')
        ax1.plot(theo_lengths, theo_periods, '-', label='理论模型')
        ax1.set_xlabel('摆长 (m)')
        ax1.set_ylabel('周期 (s)')
        ax1.set_title('周期与摆长关系')
        ax1.grid(True)
        ax1.legend()
        
        # Period^2 vs Length plot (for gravity calculation)
        ax2 = fig.add_subplot(222)
        ax2.plot(exp_lengths, np.array(exp_periods)**2, 'o', label='实验数据')
        ax2.plot(theo_lengths, np.array(theo_periods)**2, '-', label='理论模型')
        ax2.set_xlabel('摆长 (m)')
        ax2.set_ylabel('周期平方 (s²)')
        ax2.set_title('周期平方与摆长关系')
        ax2.grid(True)
        ax2.legend()
        
        # Error analysis
        metrics = self.calculate_error_metrics()
        percent_errors = metrics['percent_errors']
        
        # Error vs Length plot
        ax3 = fig.add_subplot(223)
        ax3.plot(exp_lengths, percent_errors, 'o-')
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.set_xlabel('摆长 (m)')
        ax3.set_ylabel('误差百分比 (%)')
        ax3.set_title('测量误差分析')
        ax3.grid(True)
        
        # Error distribution
        ax4 = fig.add_subplot(224)
        ax4.hist(percent_errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=np.mean(percent_errors), color='r', linestyle='--', label=f'平均误差: {np.mean(percent_errors):.2f}%')
        ax4.set_xlabel('误差百分比 (%)')
        ax4.set_ylabel('频次')
        ax4.set_title('误差分布')
        ax4.grid(True)
        ax4.legend()
        
        # Error metrics as text
        textstr = '\n'.join([
            f"平均绝对误差: {metrics['mae']:.4f} s",
            f"均方根误差: {metrics['rmse']:.4f} s",
            f"平均相对误差: {metrics['mape']:.2f}%",
            f"最大误差: {metrics['max_error']:.4f} s",
            f"误差标准差: {metrics['error_std']:.4f} s"
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
        
        return fig
        
    def pendulum_length_vs_period_experiment(self, pendulum_class, lengths, gravity=9.8, mass=0.1, 
                                             damping=0.1, initial_angle=np.pi/12, t_span=(0, 20), 
                                             t_points=1000):
        """
        Run pendulum experiment with varying lengths
        
        Parameters:
        pendulum_class: pendulum simulation class
        lengths: array of lengths to test
        gravity: gravity acceleration
        mass: pendulum mass
        damping: damping coefficient
        initial_angle: initial angle
        t_span: simulation time span
        t_points: number of simulation points
        
        Returns:
        results: dictionary containing experiment results
        fig: matplotlib figure object
        """
        periods = []
        periods_theoretical = []
        
        # Run simulations for each length
        for length in lengths:
            # Create pendulum instance
            pendulum = pendulum_class(length=length, mass=mass, gravity=gravity, 
                                     damping=damping, initial_angle=initial_angle)
            
            # 计算duration和time_step
            duration = t_span[1] - t_span[0]
            time_step = duration / t_points
            
            # Run simulation
            pendulum.simulate(duration=duration, time_step=time_step)
            
            # Calculate period
            _, period = pendulum.calculate_periods()
            periods.append(period)
            
            # Calculate theoretical period (small angles)
            period_theoretical = 2 * np.pi * np.sqrt(length / gravity)
            periods_theoretical.append(period_theoretical)
        
        # Calculate g from measured periods
        g_estimate, g_uncertainty = self.fit_period_length_relation(lengths, np.array(periods))
        
        # Store results
        results = {
            'lengths': lengths.tolist() if isinstance(lengths, np.ndarray) else lengths,
            'periods': periods,
            'periods_theoretical': periods_theoretical,
            'gravity_estimate': {
                'g_value': g_estimate,
                'g_uncertainty': g_uncertainty
            }
        }
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # T vs L plot
        ax1 = fig.add_subplot(221)
        ax1.plot(lengths, periods, 'o', label='模拟测量')
        ax1.plot(lengths, periods_theoretical, '-', label='理论值')
        ax1.set_xlabel('摆长 (m)')
        ax1.set_ylabel('周期 (s)')
        ax1.set_title('周期与摆长关系')
        ax1.grid(True)
        ax1.legend()
        
        # T^2 vs L plot
        ax2 = fig.add_subplot(222)
        periods_squared = np.array(periods)**2
        periods_theoretical_squared = np.array(periods_theoretical)**2
        
        ax2.plot(lengths, periods_squared, 'o', label='模拟测量')
        ax2.plot(lengths, periods_theoretical_squared, '-', label='理论值')
        
        # Add fitted line
        fit_line = 4 * np.pi**2 / g_estimate * np.array(lengths)
        ax2.plot(lengths, fit_line, '--', label=f'拟合: g = {g_estimate:.4f} m/s²')
        
        ax2.set_xlabel('摆长 (m)')
        ax2.set_ylabel('周期平方 (s²)')
        ax2.set_title('周期平方与摆长关系')
        ax2.grid(True)
        ax2.legend()
        
        # Percent error plot
        ax3 = fig.add_subplot(223)
        percent_errors = [(p - pt) / pt * 100 for p, pt in zip(periods, periods_theoretical)]
        ax3.plot(lengths, percent_errors, 'o-')
        ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        ax3.set_xlabel('摆长 (m)')
        ax3.set_ylabel('误差百分比 (%)')
        ax3.set_title('测量误差分析')
        ax3.grid(True)
        
        # g-value plot
        ax4 = fig.add_subplot(224)
        g_values = []
        for l, p in zip(lengths, periods):
            g_val = 4 * np.pi**2 * l / (p**2)
            g_values.append(g_val)
            
        ax4.plot(lengths, g_values, 'o-', label='各测量点计算的g值')
        ax4.axhline(y=gravity, color='r', linestyle='-', label=f'真实值: g = {gravity} m/s²')
        ax4.axhline(y=g_estimate, color='g', linestyle='--', label=f'拟合值: g = {g_estimate:.4f} m/s²')
        ax4.fill_between(lengths, g_estimate - g_uncertainty, g_estimate + g_uncertainty, 
                         color='g', alpha=0.2, label=f'不确定度: ±{g_uncertainty:.4f} m/s²')
        ax4.set_xlabel('摆长 (m)')
        ax4.set_ylabel('重力加速度 (m/s²)')
        ax4.set_title('重力加速度测量')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        
        return results, fig
        
    def analyze_controlled_experiment(self, control_param, param_values, results_dict):
        """
        Analyze experiment with controlled parameter variation
        
        Parameters:
        control_param: name of controlled parameter
        param_values: list of parameter values
        results_dict: dictionary of results for each parameter value
        
        Returns:
        analysis: analysis results dictionary
        fig: matplotlib figure object
        """
        periods = []
        g_values = []
        damping_coeffs = []
        damping_uncerts = []
        
        for param_val in param_values:
            result = results_dict[str(param_val)]
            periods.append(result.get('avg_period', 0))
            
            # Extract g values
            g_val = result.get('gravity', {}).get('value', 0)
            g_values.append(g_val)
            
            # Extract damping coefficients
            damping = result.get('damping', {})
            damping_coeffs.append(damping.get('coefficient', 0))
            damping_uncerts.append(damping.get('uncertainty', 0))
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            control_param: param_values,
            'avg_period': periods,
            'g_value': g_values,
            'damping_coefficient': damping_coeffs,
            'damping_uncertainty': damping_uncerts
        })
        
        # Create visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Period vs Parameter
        ax1 = fig.add_subplot(221)
        ax1.plot(param_values, periods, 'o-')
        ax1.set_xlabel(control_param)
        ax1.set_ylabel('平均周期 (s)')
        ax1.set_title(f'周期与{control_param}关系')
        ax1.grid(True)
        
        # g value vs Parameter
        ax2 = fig.add_subplot(222)
        ax2.plot(param_values, g_values, 'o-')
        ax2.set_xlabel(control_param)
        ax2.set_ylabel('重力加速度 (m/s²)')
        ax2.set_title(f'重力加速度与{control_param}关系')
        ax2.grid(True)
        
        # Damping coefficient vs Parameter
        ax3 = fig.add_subplot(223)
        ax3.errorbar(param_values, damping_coeffs, yerr=damping_uncerts, fmt='o-')
        ax3.set_xlabel(control_param)
        ax3.set_ylabel('阻尼系数')
        ax3.set_title(f'阻尼系数与{control_param}关系')
        ax3.grid(True)
        
        # All metrics normalized
        ax4 = fig.add_subplot(224)
        
        # Normalize each metric to its maximum value
        periods_norm = np.array(periods) / np.max(periods)
        g_values_norm = np.array(g_values) / np.max(g_values)
        damping_norm = np.array(damping_coeffs) / np.max(damping_coeffs) if np.max(damping_coeffs) != 0 else np.zeros_like(damping_coeffs)
        
        ax4.plot(param_values, periods_norm, 'o-', label='归一化周期')
        ax4.plot(param_values, g_values_norm, 's-', label='归一化g值')
        ax4.plot(param_values, damping_norm, '^-', label='归一化阻尼系数')
        ax4.set_xlabel(control_param)
        ax4.set_ylabel('归一化值')
        ax4.set_title('归一化参数对比')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        
        return analysis_df, fig 