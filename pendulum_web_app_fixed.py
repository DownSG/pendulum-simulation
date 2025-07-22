import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer
import base64
from io import BytesIO  # 仍需用于CSV导出

# 配置matplotlib支持中文显示
import matplotlib
# Windows系统使用微软雅黑字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 添加辅助函数确保每个图表都正确设置中文字体
def setup_chinese_font(fig=None, ax=None):
    """配置图表支持中文字体"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    if fig is not None:
        for text_obj in fig.findobj(matplotlib.text.Text):
            text_obj.set_fontfamily(['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif'])
    
    if ax is not None:
        for text in ax.texts:
            text.set_fontfamily(['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif'])
        
        ax.set_title(ax.get_title(), fontfamily='Microsoft YaHei')
        ax.set_xlabel(ax.get_xlabel(), fontfamily='Microsoft YaHei')
        ax.set_ylabel(ax.get_ylabel(), fontfamily='Microsoft YaHei')

st.set_page_config(page_title="单摆精确测量可视化平台", layout="wide")

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

# 辅助函数：生成动画GIF
def create_pendulum_animation(pendulum, duration=5, fps=20):
    """
    创建单摆运动的静态图像序列，而不是动画
    """
    # 创建图表并明确指定字体以支持中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5*pendulum.length, 1.5*pendulum.length)
    ax.set_ylim(-1.5*pendulum.length, 0.5*pendulum.length)
    ax.grid(True)
    
    # 绘制轨迹
    ax.plot(pendulum.simulation_results['x_position'], 
            pendulum.simulation_results['y_position'], 
            'r-', alpha=0.5, label=u'轨迹')
    
    # 选择当前位置点绘制单摆
    frame_idx = len(pendulum.simulation_results['time']) // 3  # 选择中间位置
    x = pendulum.simulation_results['x_position'][frame_idx]
    y = pendulum.simulation_results['y_position'][frame_idx]
    
    # 绘制摆点和摆球
    ax.plot([0], [0], 'ko', markersize=8)  # 摆点
    ax.plot([0, x], [0, y], 'k-', linewidth=2)  # 摆杆
    ax.plot([x], [y], 'ro', markersize=10)  # 摆球
    
    # 添加标题和信息 - 使用unicode字符串确保中文正确显示
    ax.set_title(u'单摆运动模拟')
    ax.text(0.02, 0.95, u'时间: {:.2f} s'.format(pendulum.simulation_results["time"][frame_idx]), 
            transform=ax.transAxes)
    ax.text(0.02, 0.90, u'周期: {:.4f} s'.format(2 * np.pi * np.sqrt(pendulum.length / pendulum.gravity)), 
            transform=ax.transAxes)
    
    ax.legend()
    fig.tight_layout()
    
    # 返回图表而不是保存为GIF
    return fig

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
        # 创建并显示动画
        with st.spinner("生成动画..."):
            animation_fig = create_pendulum_animation(pendulum)
            st.pyplot(animation_fig)
    
    with tab2:
        # 生成可视化
        with st.spinner("生成图表..."):
            fig = pendulum.visualize()
            # 确保图表中文字体显示正确
            setup_chinese_font(fig=fig)
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
        # 确保图表中文字体显示正确
        setup_chinese_font(fig=fig)
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
    # 确保图表中文字体显示正确
    setup_chinese_font(fig=fig)
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