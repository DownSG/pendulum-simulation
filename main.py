import numpy as np
import matplotlib.pyplot as plt
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer

# 尝试设置matplotlib后端，如果失败则不设置
try:
    import matplotlib
    matplotlib.use('TkAgg')  # 设置后端，确保动画能正确显示
except:
    print("警告: 无法设置matplotlib后端，可能影响动画显示")

# 尝试导入IPython相关模块，不是必需的
try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    print("提示: IPython不可用，某些功能可能受限")
    IPYTHON_AVAILABLE = False

def run_pendulum_simulation():
    """
    运行单摆模拟实验
    """
    print("=== 单摆精确测量虚拟平台 ===")
    
    # 创建单摆实例
    pendulum = PendulumSimulation(
        length=1.0,          # 摆长 (m)
        mass=0.1,            # 质量 (kg)
        gravity=9.8,         # 重力加速度 (m/s^2)
        damping=0.1,         # 阻尼系数
        initial_angle=np.pi/6  # 初始角度 (30度)
    )
    
    # 运行模拟
    print("\n运行单摆模拟...")
    results = pendulum.simulate(t_span=(0, 20), t_points=1000)
    
    # 计算周期
    periods, avg_period = pendulum.calculate_periods()
    print(f"测量的平均周期: {avg_period:.6f} s")
    
    # 计算重力加速度
    g = pendulum.calculate_gravity()
    print(f"根据周期计算的重力加速度: {g:.6f} m/s^2")
    
    # 可视化结果
    print("\n生成图表...")
    fig = pendulum.visualize()
    plt.show()
    
    # 创建动画
    print("\n创建单摆动画...")
    anim = pendulum.animate_pendulum(interval=50)
    
    # 返回结果
    return pendulum, results, anim

def compare_theory_with_noise():
    """
    比较理论模型与含噪声的"实验"数据
    """
    print("\n=== 理论与实验数据对比 ===")
    
    # 创建无阻尼理论模型
    theory = PendulumSimulation(
        length=1.0,
        mass=0.1,
        gravity=9.8,
        damping=0.01,  # 极小阻尼
        initial_angle=np.pi/6
    )
    
    # 运行理论模拟
    print("\n运行理论模型...")
    theory_results = theory.simulate(t_span=(0, 10), t_points=500)
    
    # 创建带阻尼和噪声的"实验"模型
    experiment = PendulumSimulation(
        length=1.0,
        mass=0.1,
        gravity=9.81,  # 略微不同的重力加速度
        damping=0.15,  # 更大的阻尼
        initial_angle=np.pi/6 + 0.02  # 初始角度有小误差
    )
    
    # 运行实验模拟
    print("\n运行\"实验\"模型...")
    exp_results = experiment.simulate(t_span=(0, 10), t_points=500)
    
    # 添加测量噪声
    print("\n添加测量噪声...")
    np.random.seed(42)  # 设置随机种子以便结果可重现
    noise_level = 0.005  # 噪声水平
    
    for key in ['angle', 'x_position', 'y_position']:
        noise = np.random.normal(0, noise_level, len(exp_results[key]))
        exp_results[key] = exp_results[key] + noise
    
    # 重新计算派生数据
    exp_results['angular_velocity'] = np.gradient(exp_results['angle'], exp_results['time'])
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    analyzer.add_theoretical_data(theory_results)
    analyzer.add_experimental_data(exp_results)
    
    # 计算误差指标
    print("\n计算误差指标...")
    error_metrics = analyzer.calculate_error_metrics()
    print("误差指标:")
    for key, value in error_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 可视化比较
    print("\n生成对比图表...")
    fig = analyzer.visualize_comparison()
    plt.show()
    
    # 导出数据到CSV
    analyzer.export_data_to_csv('pendulum_data_comparison.csv')
    
    return analyzer

def measure_gravity_experiment():
    """
    通过不同摆长测量重力加速度的实验
    """
    print("\n=== 重力加速度测量实验 ===")
    
    # 准备不同的摆长
    lengths = np.linspace(0.5, 2.0, 10)  # 10个不同长度的摆
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    
    # 运行摆长周期实验
    print("\n运行不同摆长的周期测量...")
    results, fig = analyzer.pendulum_length_vs_period_experiment(
        PendulumSimulation,
        lengths,
        gravity=9.8,
        mass=0.1,
        damping=0.05,
        initial_angle=np.pi/12,
        t_span=(0, 20),
        t_points=1000
    )
    
    # 显示结果
    print(f"\n实验测量的重力加速度: {results['gravity_estimate']['g_value']:.6f} ± {results['gravity_estimate']['g_uncertainty']:.6f} m/s^2")
    print(f"相对于标准值 (9.80665 m/s^2) 的误差: {abs(results['gravity_estimate']['g_value'] - 9.80665)/9.80665*100:.6f}%")
    
    # 显示图表
    plt.show()
    
    return results

def interactive_mode():
    """
    交互式使用单摆虚拟平台
    """
    if not IPYTHON_AVAILABLE:
        print("错误: 交互式模式需要IPython环境 (如Jupyter Notebook)")
        return
    
    try:
        import ipywidgets as widgets
    except ImportError:
        print("错误: ipywidgets 不可用，无法使用交互式模式")
        return
    
    # 参数范围
    length_slider = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='摆长 (m):')
    mass_slider = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description='质量 (kg):')
    gravity_slider = widgets.FloatSlider(value=9.8, min=1.0, max=20.0, step=0.1, description='重力加速度:')
    damping_slider = widgets.FloatSlider(value=0.1, min=0.0, max=1.0, step=0.01, description='阻尼系数:')
    angle_slider = widgets.FloatSlider(value=30, min=5, max=90, step=5, description='初始角度 (°):')
    
    # 输出控件
    output = widgets.Output()
    
    # 运行按钮
    run_button = widgets.Button(description="运行模拟")
    
    def on_run_button_click(b):
        with output:
            output.clear_output()
            
            # 创建单摆
            pendulum = PendulumSimulation(
                length=length_slider.value,
                mass=mass_slider.value,
                gravity=gravity_slider.value,
                damping=damping_slider.value,
                initial_angle=np.deg2rad(angle_slider.value)  # 转换为弧度
            )
            
            # 运行模拟
            print("运行模拟中...")
            results = pendulum.simulate(t_span=(0, 20), t_points=1000)
            
            # 计算周期
            periods, avg_period = pendulum.calculate_periods()
            print(f"测量的平均周期: {avg_period:.6f} s")
            
            # 计算重力加速度
            g = pendulum.calculate_gravity()
            print(f"根据周期计算的重力加速度: {g:.6f} m/s^2")
            
            # 理论周期 (小角近似)
            theory_period = 2 * np.pi * np.sqrt(pendulum.length / pendulum.gravity)
            print(f"理论周期 (小角近似): {theory_period:.6f} s")
            print(f"相对误差: {abs(avg_period - theory_period)/theory_period*100:.4f}%")
            
            # 显示图表
            fig = pendulum.visualize()
            plt.show()
            
    # 连接按钮点击事件
    run_button.on_click(on_run_button_click)
    
    # 显示控件
    print("=== 单摆虚拟平台交互式模式 ===")
    print("调整参数并点击'运行模拟'按钮")
    
    # 创建UI布局
    ui = widgets.VBox([
        widgets.HBox([length_slider, mass_slider]),
        widgets.HBox([gravity_slider, damping_slider]),
        angle_slider,
        run_button,
        output
    ])
    
    display(ui)

if __name__ == "__main__":
    print("单摆精确测量虚拟平台")
    print("-----------------------")
    print("请选择要运行的实验：")
    print("1. 单摆模拟实验")
    print("2. 理论与带噪声实验数据对比")
    print("3. 重力加速度测量实验")
    print("4. 交互式模式 (Jupyter Notebook/IPython)")
    
    try:
        choice = input("请输入选项 (1-4): ")
        
        if choice == '1':
            pendulum, results, anim = run_pendulum_simulation()
            # 如果在支持显示HTML的环境中，显示动画
            if IPYTHON_AVAILABLE:
                try:
                    from IPython import get_ipython
                    if get_ipython() is not None:
                        display(HTML(anim.to_jshtml()))
                except Exception as e:
                    print(f"注意: 无法显示动画: {e}")
                
        elif choice == '2':
            analyzer = compare_theory_with_noise()
            
        elif choice == '3':
            results = measure_gravity_experiment()
            
        elif choice == '4':
            interactive_mode()
            
        else:
            print("无效的选项！")
            
    except KeyboardInterrupt:
        print("\n程序已中断")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc() 