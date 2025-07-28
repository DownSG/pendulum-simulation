import numpy as np
import matplotlib.pyplot as plt
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer

# Try to set matplotlib backend, if fails don't set it
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend to ensure animation displays correctly
except:
    print("警告: 无法设置 matplotlib 后端，动画显示可能会受到影响")

# Try to import IPython related modules, not required
try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    print("注意: IPython不可用，某些功能可能会受限")
    IPYTHON_AVAILABLE = False

def run_pendulum_simulation():
    """
    Run pendulum simulation experiment
    """
    print("=== 单摆精密测量平台 ===")
    
    # Create pendulum instance
    pendulum = PendulumSimulation(
        length=1.0,
        mass=0.1,
        gravity=9.8,
        damping=0.1,
        initial_angle=np.pi/6
    )
    
    # Run simulation
    print("正在进行单摆模拟...")
    pendulum.simulate(duration=10.0, time_step=0.01)
    
    # Calculate period
    print("分析结果中...")
    kinetic_energy, potential_energy, total_energy = pendulum.calculate_energy()
    
    # Display results
    print("\n--- 单摆模拟结果 ---")
    print(f"摆长: 1.0 m")
    print(f"初始角度: {np.pi/6:.4f} rad ({np.pi/6 * 180 / np.pi:.2f}°)")
    print(f"重力加速度: 9.8 m/s²")
    print(f"阻尼系数: 0.1 kg/s")
    
    # Plot results
    print("\n生成可视化结果...")
    pendulum.visualize()
    
    print("\n模拟完成!")
    
def run_gravitational_acceleration_experiment():
    """
    Run experiment to determine gravitational acceleration
    """
    print("=== 重力加速度测量实验 ===")
    
    # Create pendulum with predefined parameters
    pendulum = PendulumSimulation(
        length=1.0,
        mass=0.1,
        gravity=9.8,  # True value, but we'll "measure" it
        damping=0.05,  # Low damping for better measurement
        initial_angle=np.pi/12  # Small angle for better approximation
    )
    
    # Create data analyzer
    analyzer = DataAnalyzer()
    
    # Define lengths to test
    print("正在测试不同摆长...")
    lengths = np.linspace(0.2, 2.0, 10)
    
    # Run experiment with varying lengths
    results, fig = analyzer.pendulum_length_vs_period_experiment(
        PendulumSimulation,
        lengths,
        gravity=9.8,
        mass=0.1,
        damping=0.05,
        initial_angle=np.pi/12
    )
    
    # Display measured value
    g_value = results['gravity_estimate']['g_value']
    g_uncertainty = results['gravity_estimate']['g_uncertainty']
    
    print("\n--- 测量结果 ---")
    print(f"重力加速度: {g_value:.6f} ± {g_uncertainty:.6f} m/s²")
    print(f"相对误差: {abs(g_value - 9.8) / 9.8 * 100:.4f}%")
    
    # Display visualization
    plt.show()
    print("\n实验完成!")

def run_damping_analysis():
    """
    Analyze damping effects
    """
    print("=== 单摆阻尼效应分析 ===")
    
    # Create pendulum with predefined parameters
    pendulum = PendulumSimulation(
        length=1.0,
        mass=0.1,
        gravity=9.8,
        damping=0.1,  # Moderate damping
        initial_angle=np.pi/4  # Larger angle for clearer damping effects
    )
    
    # Run simulation for a longer time
    print("正在运行长时间模拟...")
    pendulum.simulate(duration=20.0, time_step=0.01)
    
    # Plot results
    print("\n生成阻尼效应可视化...")
    pendulum.visualize()
    
    # Create animation
    print("\n创建单摆运动动画...")
    pendulum.animate_pendulum(interval=50)
    
    print("\n分析完成!")

def interactive_menu():
    """
    Interactive menu for pendulum simulator
    """
    while True:
        print("\n=== 单摆物理实验平台 ===")
        print("1. 运行单摆模拟")
        print("2. 测量重力加速度")
        print("3. 分析阻尼效应")
        print("4. 退出")
        
        choice = input("\n请选择操作 (1-4): ")
        
        if choice == '1':
            run_pendulum_simulation()
        elif choice == '2':
            run_gravitational_acceleration_experiment()
        elif choice == '3':
            run_damping_analysis()
        elif choice == '4':
            print("感谢使用!")
            break
        else:
            print("无效选择，请重试。")

if __name__ == "__main__":
    interactive_menu() 