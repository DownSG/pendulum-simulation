import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer
import base64
from io import BytesIO
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

# 创建单帧图像
def create_pendulum_frame(x_pos, y_pos, time_val, length, gravity, frame_idx):
    """创建单摆运动的单一帧"""
    # 创建图表
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.5*length, 1.5*length)
    ax.set_ylim(-1.5*length, 0.5*length)
    ax.grid(True)
    
    # 绘制完整轨迹线（变浅，作为背景）
    ax.plot(x_pos, y_pos, 'r-', alpha=0.1, linewidth=1)
    
    # 绘制当前轨迹
    trail_start = max(0, frame_idx - 20)  # 只显示最近的20个点
    ax.plot(x_pos[trail_start:frame_idx+1], y_pos[trail_start:frame_idx+1], 'r-', alpha=0.5, linewidth=1.5)
    
    # 当前位置
    x = x_pos[frame_idx]
    y = y_pos[frame_idx]
    
    # 绘制摆点和摆杆
    ax.plot([0], [0], 'ko', markersize=8)  # 摆点
    ax.plot([0, x], [0, y], 'k-', linewidth=2)  # 摆杆
    ax.plot([x], [y], 'ro', markersize=10)  # 摆球
    
    # 添加文本
    ax.set_title('Pendulum Motion Animation')
    ax.text(0.02, 0.95, f'Time: {time_val[frame_idx]:.2f} s', transform=ax.transAxes)
    ax.text(0.02, 0.90, f'Period: {2*np.pi*np.sqrt(length/gravity):.4f} s', transform=ax.transAxes)
    
    ax.legend(['Path', 'Trail', 'Pendulum'], loc='upper right')
    fig.tight_layout()
    
    return fig

# 修改PendulumSimulation类的visualize方法中的中文标签
def apply_english_labels(fig):
    """将图表中的中文标签替换为英文标签"""
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

# 使用"缓存运行"以保存模拟结果，避免重复计算
@st.cache_data
def run_simulation_data(length, mass, gravity, damping, initial_angle_rad, t_end):
    """运行单摆模拟并返回可序列化的数据（带缓存）"""
    pendulum = PendulumSimulation(
        length=length, 
        mass=mass, 
        gravity=gravity,
        damping=damping, 
        initial_angle=initial_angle_rad
    )
    
    # 运行模拟
    results = pendulum.simulate(t_span=(0, t_end), t_points=500)
    
    # 计算周期
    periods, avg_period = pendulum.calculate_periods()
    
    # 计算重力加速度
    g_calculated = pendulum.calculate_gravity()
    
    # 只返回可序列化的数据，不返回pendulum对象
    return {
        "time": results["time"].tolist(),  # 转换numpy数组为列表以便序列化
        "angle": results["angle"].tolist(),
        "x_position": results["x_position"].tolist(),
        "y_position": results["y_position"].tolist(),
        "total_energy": results["total_energy"].tolist(),
        "periods": periods,
        "avg_period": avg_period,
        "g_calculated": g_calculated
    }

if app_mode == "单摆基本模拟":
    st.header("单摆基本模拟")
    
    # 使用缓存运行模拟
    with st.spinner("运行单摆模拟..."):
        sim_data = run_simulation_data(
            length, mass, gravity, damping, initial_angle_rad, t_end
        )
        
        # 从缓存数据中获取结果
        time_data = np.array(sim_data["time"])
        angle_data = np.array(sim_data["angle"])
        x_position = np.array(sim_data["x_position"])
        y_position = np.array(sim_data["y_position"])
        total_energy = np.array(sim_data["total_energy"])
        periods = sim_data["periods"]
        avg_period = sim_data["avg_period"]
        g_calculated = sim_data["g_calculated"]
    
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
        # 动画设置
        st.info("单摆运动动画。实时显示单摆运动情况。")
        
        # 动画控制
        st.write("**动画控制**")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            play_animation = st.button("播放动画", use_container_width=True)
        with col2:
            frame_speed = st.slider("播放速度", 1, 10, 5, 1)
        with col3:
            num_frames = st.slider("帧数", 20, 100, 50, 10)
        
        # 动画显示区域
        animation_container = st.empty()
        status_container = st.empty()
        
        # 当用户点击播放按钮时，运行动画
        if play_animation:
            # 计算帧索引
            frame_indices = np.linspace(0, len(time_data) - 1, num_frames).astype(int)
            
            # 播放动画
            for i, idx in enumerate(frame_indices):
                # 显示进度
                progress = (i + 1) / len(frame_indices)
                status_container.progress(progress, f"播放中... {i+1}/{len(frame_indices)}")
                
                # 创建帧
                fig = create_pendulum_frame(
                    x_position, y_position, time_data, length, gravity, idx
                )
                animation_container.pyplot(fig)
                plt.close(fig)  # 关闭图表以释放内存
                
                # 控制帧速率
                delay = 0.2 / frame_speed  # 基本延迟200ms，除以速度
                time.sleep(delay)
            
            # 动画完成
            status_container.success("动画播放完成！点击'播放动画'按钮重新播放。")
        else:
            # 初始帧
            fig = create_pendulum_frame(
                x_position, y_position, time_data, length, gravity, 0
            )
            animation_container.pyplot(fig)
            plt.close(fig)
            status_container.info("点击'播放动画'按钮开始播放。")
    
    with tab2:
        # 手动生成可视化图表
        with st.spinner("生成图表..."):
            # 创建新的单摆实例来获取图表
            pendulum = PendulumSimulation(
                length=length, 
                mass=mass, 
                gravity=gravity,
                damping=damping, 
                initial_angle=initial_angle_rad
            )
            # 设置模拟结果
            pendulum.simulation_results = {
                "time": time_data,
                "angle": angle_data,
                "x_position": x_position,
                "y_position": y_position,
                "total_energy": total_energy
            }
            
            # 生成图表并应用英文标签
            fig = pendulum.visualize()
            fig = apply_english_labels(fig)
            st.pyplot(fig)
    
    with tab3:
        # 显示数据
        st.subheader("模拟数据")
        data = {
            "时间 (s)": time_data,
            "角度 (rad)": angle_data,
            "X位置 (m)": x_position,
            "Y位置 (m)": y_position,
            "总能量 (J)": total_energy
        }
        
        # 仅显示部分数据点
        display_points = min(100, len(time_data))
        step = len(time_data) // display_points
        
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