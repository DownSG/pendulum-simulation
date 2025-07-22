import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class PendulumSimulation:
    """
    单摆运动的精确模拟类
    包含考虑空气阻力的非线性单摆模型
    """
    def __init__(self, length=1.0, mass=0.1, gravity=9.8, damping=0.1, initial_angle=np.pi/6):
        """
        初始化单摆模拟器
        
        参数:
        length: 摆长(米)
        mass: 摆球质量(kg)
        gravity: 重力加速度(m/s^2)
        damping: 阻尼系数，表示阻尼力与速度的比例系数
        initial_angle: 初始角度(弧度)
        """
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.damping = damping
        self.initial_angle = initial_angle
        self.simulation_results = None
        
    def pendulum_ode(self, t, y):
        """
        单摆运动的微分方程：
        d²θ/dt² + (b/mL²)·dθ/dt + (g/L)·sinθ = 0
        
        y[0]: 角度 theta
        y[1]: 角速度 omega = dθ/dt
        
        返回:
        dydt: [dθ/dt, d²θ/dt²]
        """
        theta, omega = y
        
        # 考虑空气阻力的非线性单摆方程
        # 正确的阻尼项应为: (damping/m)·L·dθ/dt
        # 因为阻尼力矩 = b·L·ω
        dydt = [
            omega,
            -self.gravity / self.length * np.sin(theta) - self.damping / self.mass * omega
        ]
        return dydt
    
    def simulate(self, t_span=(0, 10), t_points=1000):
        """
        模拟单摆运动
        
        参数:
        t_span: 时间范围(开始时间, 结束时间)
        t_points: 时间点数量
        
        返回:
        模拟结果字典
        """
        t_eval = np.linspace(t_span[0], t_span[1], t_points)
        initial_state = [self.initial_angle, 0.0]  # 初始角度和初始角速度
        
        # 使用高精度积分器求解微分方程
        solution = solve_ivp(
            self.pendulum_ode,
            t_span,
            initial_state,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10,  # 提高积分精度
            atol=1e-10   # 提高积分精度
        )
        
        # 提取结果
        times = solution.t
        angles = solution.y[0]
        angular_velocities = solution.y[1]
        
        # 计算位置坐标 (x向右, y向下为正)
        x_positions = self.length * np.sin(angles)
        y_positions = -self.length * np.cos(angles)
        
        # 计算速度
        vx = self.length * angular_velocities * np.cos(angles)
        vy = self.length * angular_velocities * np.sin(angles)
        
        # 计算总能量
        # 动能 K = 1/2·m·L²·ω²
        kinetic_energy = 0.5 * self.mass * (self.length * angular_velocities)**2
        
        # 势能 U = m·g·h = m·g·L·(1-cosθ)
        # 零势能点为摆长垂直向下的位置
        potential_energy = self.mass * self.gravity * self.length * (1 - np.cos(angles))
        
        # 总能量
        total_energy = kinetic_energy + potential_energy
        
        # 存储结果
        self.simulation_results = {
            'time': times,
            'angle': angles,
            'angular_velocity': angular_velocities,
            'x_position': x_positions,
            'y_position': y_positions,
            'velocity_x': vx,
            'velocity_y': vy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'total_energy': total_energy
        }
        
        return self.simulation_results
    
    def calculate_periods(self):
        """
        计算单摆的周期
        通过寻找角位移的峰值来精确计算
        
        返回:
        periods: 周期列表
        average_period: 平均周期
        """
        if self.simulation_results is None:
            raise ValueError("请先运行模拟!")
            
        angles = self.simulation_results['angle']
        times = self.simulation_results['time']
        
        # 寻找角度的峰值（极大值）
        peaks = []
        for i in range(1, len(angles)-1):
            if angles[i] > angles[i-1] and angles[i] > angles[i+1]:
                # 二次插值找到精确峰值时间
                t0, t1, t2 = times[i-1], times[i], times[i+1]
                y0, y1, y2 = angles[i-1], angles[i], angles[i+1]
                
                # 拟合二次函数 y = a*t^2 + b*t + c 的顶点
                denom = (t0 - t1) * (t0 - t2) * (t1 - t2)
                a = (t0 * (y1 - y2) + t1 * (y2 - y0) + t2 * (y0 - y1)) / denom
                b = (t0*t0 * (y1 - y2) + t1*t1 * (y2 - y0) + t2*t2 * (y0 - y1)) / denom
                
                # 顶点时间: t = -b/(2a)
                if a != 0:
                    peak_time = -b / (2*a)
                    # 只添加在时间范围内的峰值
                    if t0 <= peak_time <= t2:
                        peaks.append(peak_time)
                else:
                    peaks.append(t1)
        
        # 计算周期 (相邻峰值间的时间间隔)
        periods = []
        for i in range(len(peaks) - 1):
            period = peaks[i+1] - peaks[i]
            # 过滤掉明显不合理的周期值
            if 0.1 < period < 10:  # 根据物理情况设置合理范围
                periods.append(period)
            
        if len(periods) == 0:
            # 如果找不到完整周期，使用小角近似计算
            small_angle_period = 2 * np.pi * np.sqrt(self.length / self.gravity)
            return [small_angle_period], small_angle_period
            
        average_period = np.mean(periods)
        return periods, average_period
    
    def calculate_gravity(self):
        """
        根据测量的周期计算重力加速度
        使用公式: g = 4π²L/T²
        
        对于大角度，使用修正公式:
        g = 4π²L/T² · (1 - sin²(θ₀/2)/16 - ...)^(-2)
        
        返回:
        g: 计算得到的重力加速度
        """
        _, avg_period = self.calculate_periods()
        
        # 根据单摆周期公式计算重力加速度
        g_uncorrected = 4 * np.pi**2 * self.length / (avg_period**2)
        
        # 大角度修正（考虑第一项非线性修正）
        if abs(self.initial_angle) > 0.1:  # 约5.7度
            correction = (1 - np.sin(self.initial_angle/2)**2 / 16)**(-2)
            g_corrected = g_uncorrected * correction
            return g_corrected
        
        return g_uncorrected
        
    def visualize(self):
        """
        可视化单摆运动
        """
        if self.simulation_results is None:
            raise ValueError("请先运行模拟!")
        
        # 创建图表
        fig = plt.figure(figsize=(15, 10))
        grid = plt.GridSpec(3, 2, figure=fig)
        
        # 1. 角度-时间图
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.plot(self.simulation_results['time'], self.simulation_results['angle'], 'b-')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('角度 (rad)')
        ax1.set_title('角度随时间变化')
        ax1.grid(True)
        
        # 2. 位置轨迹图
        ax2 = fig.add_subplot(grid[0, 1])
        ax2.plot(self.simulation_results['x_position'], self.simulation_results['y_position'], 'r-')
        ax2.set_xlabel('X位置 (m)')
        ax2.set_ylabel('Y位置 (m)')
        ax2.set_title('单摆轨迹')
        ax2.grid(True)
        ax2.axis('equal')  # 保持坐标轴比例一致
        
        # 3. 相空间图 (角度-角速度)
        ax3 = fig.add_subplot(grid[1, 0])
        ax3.plot(self.simulation_results['angle'], self.simulation_results['angular_velocity'], 'g-')
        ax3.set_xlabel('角度 (rad)')
        ax3.set_ylabel('角速度 (rad/s)')
        ax3.set_title('相空间轨迹')
        ax3.grid(True)
        
        # 4. 能量图
        ax4 = fig.add_subplot(grid[1, 1])
        ax4.plot(self.simulation_results['time'], self.simulation_results['kinetic_energy'], 'b-', label='动能')
        ax4.plot(self.simulation_results['time'], self.simulation_results['potential_energy'], 'r-', label='势能')
        ax4.plot(self.simulation_results['time'], self.simulation_results['total_energy'], 'g-', label='总能量')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('能量 (J)')
        ax4.set_title('能量随时间变化')
        ax4.legend()
        ax4.grid(True)
        
        # 5. 摆长与周期关系
        ax5 = fig.add_subplot(grid[2, 0])
        lengths = np.linspace(0.1, 2.0, 20)
        periods = [2 * np.pi * np.sqrt(l / self.gravity) for l in lengths]
        ax5.plot(lengths, periods, 'bo-')
        ax5.scatter([self.length], [2 * np.pi * np.sqrt(self.length / self.gravity)], 
                   color='red', s=100, marker='*', label='当前设置')
        ax5.set_xlabel('摆长 (m)')
        ax5.set_ylabel('周期 (s)')
        ax5.set_title('摆长与周期关系')
        ax5.legend()
        ax5.grid(True)
        
        # 6. 周期与摆长平方关系 (用于计算g)
        ax6 = fig.add_subplot(grid[2, 1])
        lengths_squared = lengths**2
        periods_array = np.array(periods)  # 转换为numpy数组以支持平方操作
        ax6.plot(lengths, periods_array**2, 'mo-')
        ax6.set_xlabel('摆长 (m)')
        ax6.set_ylabel('周期平方 (s²)')
        ax6.set_title('验证 T² ∝ L 关系')
        ax6.grid(True)
        
        plt.tight_layout()
        return fig
    
    def animate_pendulum(self, interval=50):
        """
        创建单摆运动的动画
        
        参数:
        interval: 帧间隔(毫秒)
        """
        if self.simulation_results is None:
            raise ValueError("请先运行模拟!")
            
        try:
            from matplotlib.animation import FuncAnimation
            import matplotlib.pyplot as plt
            from IPython.display import HTML
        except ImportError:
            print("请安装所需库: matplotlib")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-1.5*self.length, 1.5*self.length)
        ax.set_ylim(-1.5*self.length, 0.5*self.length)
        ax.grid(True)
        
        # 绘制摆点和摆球
        pivot, = ax.plot([0], [0], 'ko', markersize=8)
        line, = ax.plot([], [], 'k-', linewidth=2)
        bob, = ax.plot([], [], 'ro', markersize=10)
        
        # 绘制轨迹
        trace, = ax.plot([], [], 'r-', alpha=0.3)
        trace_x = []
        trace_y = []
        
        # 文本信息显示
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        
        def init():
            line.set_data([], [])
            bob.set_data([], [])
            trace.set_data([], [])
            time_text.set_text('')
            energy_text.set_text('')
            return line, bob, trace, time_text, energy_text
        
        def update(i):
            x = self.simulation_results['x_position'][i]
            y = self.simulation_results['y_position'][i]
            
            line.set_data([0, x], [0, y])
            bob.set_data([x], [y])
            
            # 更新轨迹
            trace_x.append(x)
            trace_y.append(y)
            trace.set_data(trace_x, trace_y)
            
            # 更新文本信息
            time_text.set_text(f'时间: {self.simulation_results["time"][i]:.2f} s')
            energy_text.set_text(f'总能量: {self.simulation_results["total_energy"][i]:.4f} J')
            
            return line, bob, trace, time_text, energy_text
        
        # 创建动画
        frames = min(100, len(self.simulation_results['time']))  # 限制帧数以避免过大
        step = len(self.simulation_results['time']) // frames
        
        anim = FuncAnimation(
            fig, update, frames=range(0, len(self.simulation_results['time']), step),
            init_func=init, blit=True, interval=interval
        )
        
        plt.close()  # 防止显示静态图形
        return anim 