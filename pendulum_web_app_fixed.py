import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer
import base64
from io import BytesIO  # 仍需用于CSV导出
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import time

# 设置页面
st.set_page_config(page_title="单摆精确测量可视化平台", layout="wide")

# 页面标题
st.title("单摆精确测量可视化平台")

st.markdown("""
这个应用程序允许你模拟单摆运动，调整各种参数，并可视化结果。你可以：
- 观察单摆的运动轨迹和能量变化
- 比较理论与"实验"数据
- 测量重力加速度
""")

# 创建侧边栏用于参数调整
st.sidebar.header("参数设置")

# 侧边栏中的应用选择
app_mode = st.sidebar.selectbox(
    "选择模式", 
    ["单摆基本模拟", "理论与实验数据对比", "重力加速度测量"]
)

# 通用参数设置
with st.sidebar.expander("基础参数", expanded=True):
    length = st.slider("摆长 (m)", 0.1, 2.0, 1.0, 0.1)
    mass = st.slider("质量 (kg)", 0.01, 1.0, 0.1, 0.01)
    gravity = st.slider("重力加速度 (m/s²)", 1.0, 20.0, 9.8, 0.1)
    damping = st.slider("阻尼系数", 0.0, 1.0, 0.1, 0.01)
    initial_angle = st.slider("初始角度 (°)", 5, 90, 30, 5)
    t_end = st.slider("模拟时长 (秒)", 5, 30, 10, 1)

# 将度数转换为弧度
initial_angle_rad = np.deg2rad(initial_angle)

# 改进版动画功能 - 使用HTML5方式显示真实动画
def create_pendulum_animation_html(pendulum, duration=5):
    """
    创建单摆运动的HTML动画，使用matplotlib的Animation和HTML
    """
    # 获取数据
    time_data = pendulum.simulation_results['time']
    x_pos = pendulum.simulation_results['x_position'] 
    y_pos = pendulum.simulation_results['y_position']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5*pendulum.length, 1.5*pendulum.length)
    ax.set_ylim(-1.5*pendulum.length, 0.5*pendulum.length)
    ax.grid(True)
    
    # 绘制轨迹
    ax.plot(x_pos, y_pos, 'r-', alpha=0.3, label='Trajectory')
    
    # 创建单摆元素
    pivot, = ax.plot([0], [0], 'ko', markersize=8)  # 摆点
    line, = ax.plot([], [], 'k-', linewidth=2)  # 摆杆
    mass_point, = ax.plot([], [], 'ro', markersize=10)  # 摆球
    
    # 创建时间文本
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    period_text = ax.text(0.02, 0.90, f'Period: {2*np.pi*np.sqrt(pendulum.length/pendulum.gravity):.4f} s', 
                         transform=ax.transAxes)
    
    ax.set_title('Pendulum Motion Animation')
    ax.legend(loc='upper right')
    
    # 动画更新函数
    def update(frame):
        i = min(frame, len(time_data) - 1)  # 确保索引不超出范围
        x = x_pos[i]
        y = y_pos[i]
        
        line.set_data([0, x], [0, y])
        mass_point.set_data([x], [y])
        time_text.set_text(f'Time: {time_data[i]:.2f} s')
        
        return line, mass_point, time_text
    
    # 计算帧数 - 确保动画平滑但不要太大
    num_frames = min(100, len(time_data))
    frame_indices = np.linspace(0, len(time_data) - 1, num_frames).astype(int)
    
    # 创建动画
    anim = FuncAnimation(fig, 
                         lambda i: update(frame_indices[i]), 
                         frames=len(frame_indices), 
                         interval=duration*1000/num_frames,  # 总持续时间(毫秒)除以帧数
                         blit=True)
    
    # 转换为HTML
    plt.close(fig)  # 防止在notebook中显示静态图
    return anim.to_jshtml()

# 修改PendulumSimulation类的visualize方法中的中文标签（在实际运行时进行替换）
def modify_pendulum_visualization(pendulum):
    """修改单摆可视化方法，使用英文标签"""
    original_visualize = pendulum.visualize
    
    def new_visualize():
        fig = original_visualize()
        
        # 替换图表中的标题和标签为英文
        for ax in fig.axes:
            if "角度" in ax.get_title():
                ax.set_title("Angle vs Time")
            elif "相位" in ax.get_title():
                ax.set_title("Phase Space")
            elif "位置" in ax.get_title():
                ax.set_title("Position vs Time")
            elif "能量" in ax.get_title():
                ax.set_title("Energy vs Time")
            elif "周期" in ax.get_title():
                ax.set_title("Period Measurement")
            elif "验证" in ax.get_title():
                ax.set_title("T² vs L Relationship")
                
            # 替换坐标轴标签
            if "时间" in ax.get_xlabel():
                ax.set_xlabel("Time (s)")
            elif "摆长" in ax.get_xlabel():
                ax.set_xlabel("Length (m)")
            
            if "角度" in ax.get_ylabel():
                ax.set_ylabel("Angle (rad)")
            elif "角速度" in ax.get_ylabel():
                ax.set_ylabel("Angular Velocity (rad/s)")
            elif "位置" in ax.get_ylabel():
                ax.set_ylabel("Position (m)")
            elif "能量" in ax.get_ylabel():
                ax.set_ylabel("Energy (J)")
            elif "周期" in ax.get_ylabel():
                ax.set_ylabel("Period (s)")
            elif "周期平方" in ax.get_ylabel():
                ax.set_ylabel("Period² (s²)")
        
        return fig
    
    pendulum.visualize = new_visualize
    return pendulum

# 修改DataAnalyzer类的可视化方法
def modify_analyzer_visualization(analyzer):
    """修改数据分析器可视化方法，使用英文标签"""
    original_visualize_comparison = analyzer.visualize_comparison
    
    def new_visualize_comparison():
        fig = original_visualize_comparison()
        
        # 替换图表中的标题和标签为英文
        for ax in fig.axes:
            if "角度" in ax.get_title():
                ax.set_title("Angle vs Time")
            elif "相位" in ax.get_title():
                ax.set_title("Phase Space")
            elif "位置" in ax.get_title():
                ax.set_title("Position vs Time")
            elif "能量" in ax.get_title():
                ax.set_title("Energy vs Time")
                
            # 替换坐标轴标签
            if "时间" in ax.get_xlabel():
                ax.set_xlabel("Time (s)")
            
            if "角度" in ax.get_ylabel():
                ax.set_ylabel("Angle (rad)")
            elif "角速度" in ax.get_ylabel():
                ax.set_ylabel("Angular Velocity (rad/s)")
            elif "位置" in ax.get_ylabel():
                ax.set_ylabel("Position (m)")
            elif "能量" in ax.get_ylabel():
                ax.set_ylabel("Energy (J)")
            
            # 替换图例
            if ax.get_legend():
                for text in ax.get_legend().texts:
                    if "理论" in text.get_text():
                        text.set_text("Theory")
                    elif "实验" in text.get_text():
                        text.set_text("Experiment")
        
        return fig
    
    analyzer.visualize_comparison = new_visualize_comparison
    return analyzer

if app_mode == "单摆基本模拟":
    st.header("单摆基本模拟")
    
    # 创建单摆实例
    pendulum = PendulumSimulation(
        length=length, 
        mass=mass, 
        gravity=gravity,
        damping=damping, 
        initial_angle=initial_angle_rad
    )
    
    # 修改可视化方法
    pendulum = modify_pendulum_visualization(pendulum)
    
    # 使用会话状态跟踪参数变化
    session_state_key = f"pendulum_{length}_{mass}_{gravity}_{damping}_{initial_angle}_{t_end}"
    
    # 运行模拟
    with st.spinner("运行单摆模拟..."):
        results = pendulum.simulate(t_span=(0, t_end), t_points=500)
    
    # 计算周期
    periods, avg_period = pendulum.calculate_periods()
    
    # 计算重力加速度
    g_calculated = pendulum.calculate_gravity()
    
    # 显示关键结果
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("测量的平均周期", f"{avg_period:.6f} 秒")
    with col2:
        st.metric("计算的重力加速度", f"{g_calculated:.6f} m/s²")
    with col3:
        theory_period = 2 * np.pi * np.sqrt(length / gravity)
        error = abs(avg_period - theory_period)/theory_period*100
        st.metric("理论周期误差", f"{error:.4f}%")
    
    # 创建图表选项卡
    tab1, tab2, tab3 = st.tabs(["动画", "图表", "数据"])
    
    with tab1:
        # 显示帮助信息
        st.info("调整左侧的参数，动画会实时更新以显示单摆运动。")
        
        # 创建并显示动画
        with st.spinner("生成动画..."):
            animation_html = create_pendulum_animation_html(pendulum)
            st.components.v1.html(animation_html, height=650)
    
    with tab2:
        # 生成可视化
        with st.spinner("生成图表..."):
            fig = pendulum.visualize()
            st.pyplot(fig)
    
    with tab3:
        # 显示数据
        st.subheader("模拟数据")
        data = {
            "时间 (s)": results["time"],
            "角度 (rad)": results["angle"],
            "X位置 (m)": results["x_position"],
            "Y位置 (m)": results["y_position"],
            "总能量 (J)": results["total_energy"]
        }
        
        # 仅显示部分数据点
        display_points = min(100, len(results["time"]))
        step = len(results["time"]) // display_points
        
        for key in data:
            data[key] = data[key][::step]
            
        st.dataframe(data)
        
        # 添加下载按钮
        csv = np.column_stack([data["时间 (s)"], data["角度 (rad)"], data["X位置 (m)"], data["Y位置 (m)"], data["总能量 (J)"]])
        
        # 使用pandas导出为CSV，以处理中文编码问题
        import pandas as pd
        csv_df = pd.DataFrame({
            'time': data["时间 (s)"], 
            'angle': data["角度 (rad)"], 
            'x_pos': data["X位置 (m)"], 
            'y_pos': data["Y位置 (m)"], 
            'energy': data["总能量 (J)"]
        })
        
        # 转换为CSV字符串
        csv_buffer = BytesIO()
        csv_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        st.download_button(
            label="下载数据为CSV",
            data=csv_buffer,
            file_name="pendulum_simulation_data.csv",
            mime="text/csv"
        )

elif app_mode == "理论与实验数据对比":
    st.header("理论与实验数据对比")
    
    # 添加额外的实验参数
    with st.sidebar.expander("实验参数", expanded=True):
        exp_gravity = st.slider("实验重力加速度 (m/s²)", 9.5, 10.0, 9.81, 0.01)
        exp_damping = st.slider("实验阻尼系数", 0.05, 0.5, 0.15, 0.01)
        exp_angle_error = st.slider("初始角度误差 (°)", -5.0, 5.0, 1.0, 0.5)
        noise_level = st.slider("噪声水平", 0.001, 0.02, 0.005, 0.001)
    
    # 转换为弧度
    exp_angle_error_rad = np.deg2rad(exp_angle_error)
    
    # 创建理论模型
    with st.spinner("运行理论模型..."):
        theory = PendulumSimulation(
            length=length,
            mass=mass,
            gravity=gravity,
            damping=0.01,  # 极小阻尼
            initial_angle=initial_angle_rad
        )
        theory_results = theory.simulate(t_span=(0, t_end), t_points=500)
    
    # 创建实验模型
    with st.spinner("运行实验模型..."):
        experiment = PendulumSimulation(
            length=length,
            mass=mass,
            gravity=exp_gravity,
            damping=exp_damping,
            initial_angle=initial_angle_rad + exp_angle_error_rad
        )
        exp_results = experiment.simulate(t_span=(0, t_end), t_points=500)
    
    # 添加噪声
    with st.spinner("添加测量噪声..."):
        np.random.seed(42)
        for key in ['angle', 'x_position', 'y_position']:
            noise = np.random.normal(0, noise_level, len(exp_results[key]))
            exp_results[key] = exp_results[key] + noise
        
        # 重新计算派生数据
        exp_results['angular_velocity'] = np.gradient(exp_results['angle'], exp_results['time'])
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    analyzer.add_theoretical_data(theory_results)
    analyzer.add_experimental_data(exp_results)
    
    # 修改可视化方法
    analyzer = modify_analyzer_visualization(analyzer)
    
    # 计算误差指标
    with st.spinner("计算误差指标..."):
        error_metrics = analyzer.calculate_error_metrics()
    
    # 显示误差指标
    st.subheader("误差指标")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("均方根误差 (RMSE)", f"{error_metrics['RMSE']:.6f}")
    with col2:
        st.metric("平均绝对误差 (MAE)", f"{error_metrics['MAE']:.6f}")
    with col3:
        st.metric("平均绝对百分比误差", f"{error_metrics['MAPE']:.2f}%")
    
    # 可视化比较
    st.subheader("数据对比可视化")
    with st.spinner("生成对比图表..."):
        fig = analyzer.visualize_comparison()
        st.pyplot(fig)
    
    # 保存数据
    if st.button("保存对比数据为CSV"):
        analyzer.export_data_to_csv('pendulum_data_comparison.csv')
        st.success("数据已保存到 pendulum_data_comparison.csv")

elif app_mode == "重力加速度测量实验":
    st.header("重力加速度测量实验")
    
    with st.sidebar.expander("实验参数", expanded=True):
        min_length = st.slider("最小摆长 (m)", 0.1, 1.0, 0.5, 0.1)
        max_length = st.slider("最大摆长 (m)", 1.0, 3.0, 2.0, 0.1)
        num_points = st.slider("测量点数", 3, 20, 10, 1)
        exp_gravity = st.slider("真实重力加速度 (m/s²)", 9.5, 10.0, 9.8, 0.01)
        exp_damping = st.slider("实验阻尼系数", 0.01, 0.2, 0.05, 0.01)
    
    # 准备不同的摆长
    lengths = np.linspace(min_length, max_length, num_points)
    
    # 创建数据分析器
    analyzer = DataAnalyzer()
    
    # 运行摆长周期实验
    with st.spinner("运行不同摆长的周期测量..."):
        results, fig = analyzer.pendulum_length_vs_period_experiment(
            PendulumSimulation,
            lengths,
            gravity=exp_gravity,
            mass=mass,
            damping=exp_damping,
            initial_angle=np.pi/12,
            t_span=(0, 20),
            t_points=1000
        )
    
    # 修改图表中的中文标签为英文
    for ax in fig.axes:
        if "周期" in ax.get_title():
            ax.set_title("Period vs Length")
        elif "验证" in ax.get_title():
            ax.set_title("T² vs L Relationship")
            
        if "摆长" in ax.get_xlabel():
            ax.set_xlabel("Length (m)")
        
        if "周期" in ax.get_ylabel():
            ax.set_ylabel("Period (s)")
        elif "周期平方" in ax.get_ylabel():
            ax.set_ylabel("Period² (s²)")
    
    # 显示结果
    st.subheader("测量结果")
    g_value = results['gravity_estimate']['g_value']
    g_uncertainty = results['gravity_estimate']['g_uncertainty']
    standard_g = 9.80665
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("测量重力加速度", f"{g_value:.6f} m/s²")
    with col2:
        st.metric("测量不确定度", f"±{g_uncertainty:.6f} m/s²")
    with col3:
        relative_error = abs(g_value - standard_g) / standard_g * 100
        st.metric("相对标准值误差", f"{relative_error:.6f}%")
    
    # 显示图表
    st.pyplot(fig)
    
    # 显示详细数据
    st.subheader("测量数据")
    data = {
        "摆长 (m)": lengths,
        "周期 (s)": results["periods"],
        "周期平方 (s²)": np.square(results["periods"])
    }
    st.dataframe(data)
    
    # 拟合参数
    st.subheader("线性拟合参数")
    m, b = results['gravity_estimate']['fit_params']
    st.write(f"拟合方程: T² = {m:.6f} × L + {b:.6f}")
    st.write(f"理论斜率: 4π²/g = {4 * np.pi**2 / exp_gravity:.6f}")
    st.write(f"测量斜率: {m:.6f}")
    st.write(f"理论截距: 0 (理想情况)")
    st.write(f"测量截距: {b:.6f}")

# 添加说明和注意事项
with st.expander("使用说明"):
    st.markdown("""
    ### 如何使用这个应用
    1. 在左侧栏选择模拟模式
    2. 调整各种参数以观察其影响
    3. 查看图表、数据和动画了解单摆运动规律
    
    ### 参数解释
    - **摆长**: 从支点到摆球中心的距离
    - **质量**: 摆球质量，影响能量但不影响周期
    - **重力加速度**: 影响单摆周期的主要参数
    - **阻尼系数**: 模拟空气阻力等阻尼效应
    - **初始角度**: 释放单摆时的初始角度
    
    ### 实验背景
    单摆是经典物理中研究周期运动和重力测量的重要实验。通过测量不同摆长下的周期，
    可以根据公式 T = 2π√(L/g) 精确计算重力加速度。
    """)

st.sidebar.info("© 2024 单摆精确测量虚拟平台") 