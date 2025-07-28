"""
pendulum_simulation.py
----------------------
本模块为单摆实验的本地端核心仿真与AI分析平台。
支持批量仿真、g值自动测量、阻尼分析、误差分析、敏感性分析等科研级实验物理数据处理。
所有功能均可在命令行、Jupyter Notebook等环境下直接调用，便于教学、科研和AI集成。
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd # Added for run_full_experiment

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class PendulumSimulation:
    """
    Single Pendulum Simulation Class
    Contains non-linear pendulum model with air resistance
    Supports optional PID angle control
    """
    def __init__(self, length=1.0, mass=0.1, gravity=9.8, damping=0.1, initial_angle=np.pi/6,
                 use_pid=False, pid_kp=1000, pid_kd=500, target_angle=np.pi/4):
        """
        Initialize pendulum simulator
        
        Parameters:
        length: pendulum length (m)
        mass: bob mass (kg)
        gravity: gravitational acceleration (m/s^2)
        damping: damping coefficient, proportional to velocity
        initial_angle: initial angle (rad)
        use_pid: whether to enable PID angle control
        pid_kp: PID proportional gain
        pid_kd: PID derivative gain
        target_angle: desired angle setpoint for PID controller
        """
        # Pendulum physical parameters
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.damping = damping
        
        # Initial state [angle, angular velocity]
        self.initial_state = [initial_angle, 0.0]  
        
        # PID controller parameters
        self.use_pid = use_pid
        self.pid_kp = pid_kp
        self.pid_kd = pid_kd
        self.target_angle = target_angle

        # Simulation results storage
        self.times = None
        self.angles = None
        self.angular_velocities = None
        self.control_torques = None
        
    def equations_of_motion(self, t, y):
        """
        Pendulum equations of motion
        
        Parameters:
        t: time variable (required by solver)
        y: state vector [angle, angular_velocity]
        
        Returns:
        dy/dt: state derivative [angular_velocity, angular_acceleration]
        """
        angle, angular_velocity = y
        
        # Nonlinear pendulum equation with damping
        angular_acceleration = (-self.gravity / self.length * np.sin(angle) 
                              - self.damping * angular_velocity / self.mass)
        
        # Add control torque if PID is enabled
        if self.use_pid:
            # PID control (P and D terms only)
            angle_error = self.target_angle - angle
            control_torque = self.pid_kp * angle_error - self.pid_kd * angular_velocity
            angular_acceleration += control_torque / (self.mass * self.length**2)
            
        return [angular_velocity, angular_acceleration]

    def simulate(self, duration=10.0, time_step=0.01):
        """
        Run pendulum simulation
        
        Parameters:
        duration: simulation duration (seconds)
        time_step: time step for result sampling (seconds)
        
        Returns:
        times: array of time points
        angles: array of pendulum angles
        angular_velocities: array of angular velocities
        control_torques: array of control torques (if PID is enabled)
        """
        # Time points for simulation
        t_eval = np.arange(0, duration, time_step)
        
        # Solve ODE
        solution = solve_ivp(
            self.equations_of_motion,
            [0, duration],
            self.initial_state,
            method='RK45',
            t_eval=t_eval
        )
        
        # Extract results
        self.times = solution.t
        self.angles = solution.y[0]
        self.angular_velocities = solution.y[1]
        
        # Calculate control torques if PID is enabled
        if self.use_pid:
            self.control_torques = np.zeros_like(self.times)
            for i in range(len(self.times)):
                angle_error = self.target_angle - self.angles[i]
                self.control_torques[i] = (self.pid_kp * angle_error - 
                                          self.pid_kd * self.angular_velocities[i])
        
        return self.times, self.angles, self.angular_velocities, self.control_torques
    
    def get_position(self, angles=None):
        """
        Calculate pendulum bob position from angles
        
        Parameters:
        angles: array of pendulum angles (if None, use stored angles)
        
        Returns:
        x_positions: array of x-coordinates
        y_positions: array of y-coordinates
        """
        if angles is None:
            if self.angles is None:
                raise ValueError("No angles available. Run simulate() first.")
            angles = self.angles
            
        x_positions = self.length * np.sin(angles)
        y_positions = -self.length * np.cos(angles)
        
        return x_positions, y_positions
    
    def calculate_energy(self, angles=None, angular_velocities=None):
        """
        Calculate pendulum energy (kinetic and potential)
        
        Parameters:
        angles: array of pendulum angles (if None, use stored angles)
        angular_velocities: array of angular velocities (if None, use stored velocities)
        
        Returns:
        kinetic_energy: array of kinetic energies
        potential_energy: array of potential energies
        total_energy: array of total energies
        """
        if angles is None or angular_velocities is None:
            if self.angles is None or self.angular_velocities is None:
                raise ValueError("No simulation data available. Run simulate() first.")
            angles = self.angles
            angular_velocities = self.angular_velocities
            
        # Calculate bob velocity components
        x_velocities = self.length * np.cos(angles) * angular_velocities
        y_velocities = self.length * np.sin(angles) * angular_velocities
        
        # Calculate velocities, potential and kinetic energy
        velocities = np.sqrt(x_velocities**2 + y_velocities**2)
        kinetic_energy = 0.5 * self.mass * velocities**2
        
        # Get heights (y_position + length to set zero at lowest point)
        _, y_positions = self.get_position(angles)
        heights = y_positions + self.length
        
        potential_energy = self.mass * self.gravity * heights
        total_energy = kinetic_energy + potential_energy
        
        return kinetic_energy, potential_energy, total_energy
    
    def visualize(self, show_path=True):
        """
        Visualize simulation results
        
        Parameters:
        show_path: whether to show pendulum path
        """
        if self.times is None or self.angles is None:
            raise ValueError("No simulation data available. Run simulate() first.")
        
        # Create figure with 4 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Angle vs Time plot
        ax1.plot(self.times, self.angles)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('角度 (rad)')
        ax1.set_title('角度随时间变化')
        ax1.grid(True)
        
        # Calculate x, y positions
        x_positions, y_positions = self.get_position()
        
        # Pendulum path plot
        ax2.plot(x_positions, y_positions, 'b-')
        if show_path:
            ax2.plot([0, x_positions[-1]], [0, y_positions[-1]], 'r--')
        ax2.set_xlabel('X位置 (m)')
        ax2.set_ylabel('Y位置 (m)')
        ax2.set_title('单摆轨迹')
        ax2.grid(True)
        ax2.axis('equal')
        
        # Phase space plot (angle vs angular velocity)
        ax3.plot(self.angles, self.angular_velocities)
        ax3.set_xlabel('角度 (rad)')
        ax3.set_ylabel('角速度 (rad/s)')
        ax3.set_title('相空间轨迹')
        ax3.grid(True)
        
        # Energy plot
        kinetic_energy, potential_energy, total_energy = self.calculate_energy()
        ax4.plot(self.times, kinetic_energy, 'r-', label='动能')
        ax4.plot(self.times, potential_energy, 'g-', label='势能')
        ax4.plot(self.times, total_energy, 'b-', label='总能量')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('能量 (J)')
        ax4.set_title('能量随时间变化')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def animate_pendulum(self, interval=50, save_animation=False, filename='pendulum_animation.gif'):
        """
        Create pendulum animation
        
        Parameters:
        interval: time between animation frames (ms)
        save_animation: whether to save animation to file
        filename: output file name if saving animation
        """
        if self.times is None or self.angles is None:
            raise ValueError("No simulation data available. Run simulate() first.")
            
        import matplotlib.animation as animation
        from matplotlib.patches import Circle
        
        # Calculate bob positions
        x_positions, y_positions = self.get_position()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set limits with some padding
        max_range = self.length * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_xlabel('X位置 (m)')
        ax.set_ylabel('Y位置 (m)')
        ax.set_title('单摆动画')
        ax.grid(True)
        
        # Create pendulum rod line
        line, = ax.plot([], [], 'k-', lw=2)
        
        # Create pendulum bob (circle)
        bob_radius = self.length * 0.05
        bob = Circle((0, 0), bob_radius, fc='r', zorder=3)
        ax.add_patch(bob)
        
        # Add pendulum pivot point
        ax.plot(0, 0, 'ko', markersize=10)
        
        # Optional: path trace (empty at first)
        if save_animation:
            path, = ax.plot([], [], 'b-', alpha=0.3)
            path_x = []
            path_y = []
        
        # Initialization function for animation
        def init():
            line.set_data([], [])
            bob.center = (0, 0)
            if save_animation:
                path.set_data([], [])
            return line, bob
        
        # Animation update function
        def update(i):
            x = x_positions[i]
            y = y_positions[i]
            
            line.set_data([0, x], [0, y])
            bob.center = (x, y)
            
            if save_animation:
                path_x.append(x)
                path_y.append(y)
                path.set_data(path_x, path_y)
                
            return line, bob
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.times),
            init_func=init, blit=True, interval=interval
        )
        
        if save_animation:
            ani.save(filename, writer='pillow', fps=1000/interval)
            
        plt.show()
    
    def run_full_experiment(self, lengths=None, duration=10.0, time_step=0.01):
        """
        Run a full pendulum experiment with varying lengths
        Returns period vs. length data
        
        Parameters:
        lengths: array of pendulum lengths to test (m)
        duration: simulation duration (seconds)
        time_step: time step for result sampling (seconds)
        
        Returns:
        df: DataFrame with length and period columns
        """
        if lengths is None:
            lengths = np.linspace(0.1, 2.0, 20)
            
        results = []
        
        for length in lengths:
            # Update pendulum length
            self.length = length
            
            # Run simulation
            self.simulate(duration, time_step)
            
            # Calculate period (use zero crossings)
            angles = self.angles
            times = self.times
            
            # Find where the pendulum crosses through equilibrium (sin(angle) = 0)
            # Using linear interpolation for better accuracy
            indices = np.where(np.diff(np.signbit(angles)))[0]
            
            if len(indices) >= 2:
                # Calculate average period from zero crossings
                periods = []
                for i in range(len(indices) - 2):
                    # Only use consecutive crossings in the same direction
                    if (i % 2) == 0:  
                        period = times[indices[i+2]] - times[indices[i]]
                        periods.append(period)
                
                if periods:
                    avg_period = np.mean(periods)
                    results.append({
                        'length': length,
                        'period': avg_period,
                        'period_theoretical': 2 * np.pi * np.sqrt(length / self.gravity)
                    })
        
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            return df
        else:
            return pd.DataFrame(columns=['length', 'period', 'period_theoretical'])

    def calculate_periods(self, angles=None, times=None):
        """
        计算单摆的周期数
        
        Parameters:
        angles: 角度数据数组（如果为None，则使用存储的角度）
        times: 时间数据数组（如果为None，则使用存储的时间）
        
        Returns:
        num_periods: 完整周期数
        period: 平均周期时间
        """
        if angles is None or times is None:
            if self.angles is None or self.times is None:
                raise ValueError("No simulation data available. Run simulate() first.")
            angles = self.angles
            times = self.times
            
        # 找到过零点（从负到正）
        zero_crossings = []
        for i in range(1, len(angles)):
            if angles[i-1] < 0 and angles[i] >= 0:
                # 线性插值过零点时间
                t0, t1 = times[i-1], times[i]
                a0, a1 = angles[i-1], angles[i]
                t_cross = t0 + (0 - a0) * (t1 - t0) / (a1 - a0)
                zero_crossings.append(t_cross)
        
        # 计算周期（每两个过零点为一个周期）
        periods = []
        for i in range(0, len(zero_crossings) - 1, 2):
            if i + 1 < len(zero_crossings):
                period = 2 * (zero_crossings[i+1] - zero_crossings[i])
                periods.append(period)
        
        # 计算平均周期
        if periods:
            self.period = np.mean(periods)
            return len(periods), self.period
        else:
            self.period = None
            return 0, None

def run_damping_analysis(
    length=1.0,
    gravity=9.8,
    mass=0.1,
    damping=0.1,
    initial_angle=np.pi/6,
    t_span=(0, 20),
    t_points=1000,
    export_csv=None,
    export_json=None
):
    """
    Simulate damping decay under a single pendulum length, automatically fit damping coefficient, output fitting results and visualization.
    Parameters:
        length: float, pendulum length
        gravity: float, gravitational acceleration
        mass: float, bob mass
        damping: float, damping coefficient
        initial_angle: float, initial angle (rad)
        t_span: tuple, simulation time range
        t_points: int, number of simulation time points
        export_csv: str, optional, export csv file name
        export_json: str, optional, export json file name
    Returns:
        fit_result: dict, damping fitting results
        fig: matplotlib.figure, result visualization figure
    """
    from data_analyzer import DataAnalyzer
    import json
    # Simulate
    pendulum = PendulumSimulation(length=length, mass=mass, gravity=gravity, damping=damping, initial_angle=initial_angle)
    results = pendulum.simulate(t_span=t_span, t_points=t_points)
    # Take each peak as amplitude
    angles = results['angle']
    times = results['time']
    peaks = []
    peak_times = []
    for i in range(1, len(angles)-1):
        if angles[i] > angles[i-1] and angles[i] > angles[i+1]:
            peaks.append(abs(angles[i]))
            peak_times.append(times[i])
    peaks = np.array(peaks)
    peak_times = np.array(peak_times)
    # Damping analysis
    analyzer = DataAnalyzer()
    fit_result = analyzer.analyze_damping(peak_times, peaks)
    # Optional export csv
    if export_csv:
        import pandas as pd
        df = pd.DataFrame({'time': peak_times, 'amplitude': peaks})
        df.to_csv(export_csv, index=False)
        print(f"Damping analysis data exported to {export_csv}")
    if export_json:
        with open(export_json, 'w', encoding='utf-8') as f:
            json.dump(fit_result, f, ensure_ascii=False, indent=2)
        print(f"Damping analysis results exported to {export_json}")
    # Visualization
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(peak_times, peaks, 'bo', label='Simulated Peak')
    # Fitting curve
    a0 = fit_result['initial_amplitude']
    gamma = fit_result['damping_coefficient']
    fit_curve = a0 * np.exp(-gamma * peak_times)
    ax.plot(peak_times, fit_curve, 'r--', label=f'Fitted: A={a0:.3f}e^(-{gamma:.3f}t)')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('振幅 (rad)')
    ax.set_title('单摆阻尼衰减拟合')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    print(f"Auto-fitted damping coefficient: gamma = {gamma:.6f} ± {fit_result['damping_uncertainty']:.6f}")
    return fit_result, fig

def run_error_analysis_and_comparison(
    experimental_data,
    theoretical_data,
    export_csv=None,
    export_json=None
):
    """
    Input experimental data and theoretical data, automatically calculate error metrics and generate comparison visualization.
    Parameters:
        experimental_data: dict, experimental data (needs time, angle, y_position, angular_velocity)
        theoretical_data: dict, theoretical data (same as above)
        export_csv: str, optional, export csv file name
        export_json: str, optional, export json file name
    Returns:
        error_metrics: dict, error metrics
        fig: matplotlib.figure, comparison visualization figure
    """
    from data_analyzer import DataAnalyzer
    import json
    analyzer = DataAnalyzer()
    analyzer.add_experimental_data(experimental_data)
    analyzer.add_theoretical_data(theoretical_data)
    error_metrics = analyzer.calculate_error_metrics()
    fig = analyzer.visualize_comparison()
    # Optional export csv
    if export_csv:
        import pandas as pd
        df = pd.DataFrame([error_metrics])
        df.to_csv(export_csv, index=False)
        print(f"Error analysis results exported to {export_csv}")
    if export_json:
        with open(export_json, 'w', encoding='utf-8') as f:
            json.dump(error_metrics, f, ensure_ascii=False, indent=2)
        print(f"Error analysis results exported to {export_json}")
    print("Error analysis metrics:")
    for k, v in error_metrics.items():
        print(f"{k}: {v}")
    return error_metrics, fig

def run_batch_error_analysis(
    exp_theo_data_pairs,
    export_csv=None,
    export_json=None
):
    """
    Batch error analysis for multiple experimental-theoretical data pairs.
    Parameters:
        exp_theo_data_pairs: list, each item is a tuple (experimental_data, theoretical_data)
        export_csv: str, optional, export csv file name
        export_json: str, optional, export json file name
    Returns:
        metrics_list: list, all error metrics lists
    """
    from data_analyzer import DataAnalyzer
    import json
    metrics_list = []
    for idx, (exp_data, theo_data) in enumerate(exp_theo_data_pairs):
        analyzer = DataAnalyzer()
        analyzer.add_experimental_data(exp_data)
        analyzer.add_theoretical_data(theo_data)
        metrics = analyzer.calculate_error_metrics()
        metrics['pair_index'] = idx
        metrics_list.append(metrics)
    # Optional export csv
    if export_csv:
        import pandas as pd
        df = pd.DataFrame(metrics_list)
        df.to_csv(export_csv, index=False)
        print(f"Batch error analysis results exported to {export_csv}")
    if export_json:
        with open(export_json, 'w', encoding='utf-8') as f:
            json.dump(metrics_list, f, ensure_ascii=False, indent=2)
        print(f"Batch error analysis results exported to {export_json}")
    print("Batch error analysis metrics:")
    for m in metrics_list:
        print(m)
    return metrics_list

def run_sensitivity_analysis(
    param_name,
    param_values,
    base_params=None,
    t_span=(0, 20),
    t_points=1000,
    export_csv=None,
    export_json=None
):
    """
    Multi-parameter sensitivity analysis.
    Parameters:
        param_name: str, parameter name to scan (e.g., 'length', 'damping', 'initial_angle' etc.)
        param_values: list/array, parameter values
        base_params: dict, baseline values for other parameters
        t_span: tuple, simulation time range
        t_points: int, number of simulation time points
        export_csv: str, optional, export csv file name
        export_json: str, optional, export json file name
    Returns:
        results_df: pandas.DataFrame, sensitivity analysis results
        fig: matplotlib.figure, result visualization figure
    """
    import pandas as pd
    import json
    from data_analyzer import DataAnalyzer
    if base_params is None:
        base_params = dict(length=1.0, mass=0.1, gravity=9.8, damping=0.1, initial_angle=np.pi/6)
    records = []
    for val in param_values:
        params = base_params.copy()
        params[param_name] = val
        pendulum = PendulumSimulation(**params)
        sim_result = pendulum.simulate(t_span=t_span, t_points=t_points)
        # Period, g value, damping coefficient
        periods, avg_period = pendulum.calculate_periods()
        g_val = pendulum.calculate_gravity()
        # Damping coefficient fitting
        angles = sim_result['angle']
        times = sim_result['time']
        peaks = []
        peak_times = []
        for i in range(1, len(angles)-1):
            if angles[i] > angles[i-1] and angles[i] > angles[i+1]:
                peaks.append(abs(angles[i]))
                peak_times.append(times[i])
        peaks = np.array(peaks)
        peak_times = np.array(peak_times)
        analyzer = DataAnalyzer()
        damping_fit = analyzer.analyze_damping(peak_times, peaks)
        records.append({
            param_name: val,
            'avg_period': avg_period,
            'g_value': g_val,
            'damping_coefficient': damping_fit['damping_coefficient'],
            'damping_uncertainty': damping_fit['damping_uncertainty']
        })
    df = pd.DataFrame(records)
    # Optional export csv
    if export_csv:
        df.to_csv(export_csv, index=False)
        print(f"Sensitivity analysis results exported to {export_csv}")
    if export_json:
        with open(export_json, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"Sensitivity analysis results exported to {export_json}")
    # Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    axes[0].plot(df[param_name], df['avg_period'], 'o-')
    axes[0].set_title('周期-参数关系')
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('平均周期 (s)')
    axes[1].plot(df[param_name], df['g_value'], 'o-')
    axes[1].set_title('g值-参数关系')
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('g值 (m/s²)')
    axes[2].errorbar(df[param_name], df['damping_coefficient'], yerr=df['damping_uncertainty'], fmt='o-')
    axes[2].set_title('阻尼系数-参数关系')
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel('阻尼系数')
    plt.tight_layout()
    return df, fig

# Example usage (can be called from command line or Notebook)
if __name__ == "__main__":
    # Example: Batch simulate 5 different lengths
    lengths = np.linspace(0.5, 1.5, 5)
    run_full_experiment(lengths, export_csv="pendulum_exp_results.csv")
    # Example: Damping analysis
    run_damping_analysis(export_csv="damping_analysis.csv")
    # Example: Error analysis and comparison
    pendulum = PendulumSimulation()
    exp_data = pendulum.simulate()
    theo_data = pendulum.simulate()
    run_error_analysis_and_comparison(exp_data, theo_data, export_csv="error_analysis.csv")
    # Example: Batch error analysis
    pairs = [(exp_data, theo_data) for _ in range(3)]
    run_batch_error_analysis(pairs, export_csv="batch_error_analysis.csv")
    # Example: Sensitivity analysis
    run_sensitivity_analysis('damping', np.linspace(0.01, 0.2, 6), export_csv="sensitivity_analysis.csv") 