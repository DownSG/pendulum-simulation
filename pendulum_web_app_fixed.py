import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # 添加Plotly支持
from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer
import base64
from io import BytesIO
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import time
import pandas as pd  # 确保pandas被导入

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

# 添加参数科学说明折叠面板
with st.sidebar.expander("参数影响详解", expanded=False):
    st.markdown("""
    ### 参数对单摆运动的科学影响

    **1. 摆长 (L)**
    - **影响周期**: $T = 2\pi \sqrt{L/g}$。摆长增加，周期增加，呈平方根关系。
    - **影响振动频率**: $f = \frac{1}{2\pi}\sqrt{g/L}$，摆长增加，振动频率减小。
    - **影响角加速度**: $\ddot{\theta} = -\frac{g}{L}\sin\theta$，摆长增加，角加速度减小。
    
    **2. 质量 (m)**
    - **对周期无影响**: 在理想单摆中，周期完全独立于质量（对比现实摆锤，其角动量与质量有关）。
    - **影响阻尼比**: $\zeta = \frac{b}{2m\omega_0L}$，质量增加，阻尼比减小，振动持续时间更长。
    - **影响能量幅度**: 质量增加，系统总能量增加（$E \propto m$），但运动特性不变。
    
    **3. 重力加速度 (g)**
    - **影响周期**: $T = 2\pi \sqrt{L/g}$。重力加速度增加，周期减小，呈反比例平方根关系。
    - **影响平衡恢复力**: $F_{恢复} = mg\sin\theta$。g增加，恢复力增加，振荡更快。
    - **影响固有频率**: $\omega_0 = \sqrt{g/L}$，重力加速度增加，固有频率增加。
    
    **4. 阻尼系数 (b)**
    - **影响振幅衰减率**: $\gamma = \frac{b}{2mL}$，阻尼系数增加，振幅衰减更快。
    - **影响能量损耗率**: $\frac{dE}{dt} = -b(\frac{d\theta}{dt})^2$，阻尼系数增加，能量损耗更快。
    - **影响系统类型**:
        * $\zeta < 1$: 欠阻尼（振荡衰减）
        * $\zeta = 1$: 临界阻尼（无振荡最快回到平衡）
        * $\zeta > 1$: 过阻尼（无振荡缓慢回到平衡）
    
    **5. 初始角度 ($\\theta_0$)**
    - **影响非线性效应**: 角度越大，非线性效应越显著：
        * 小角度（$<10°$）：$\sin\theta \approx \theta$，近似为简谐运动
        * 大角度：需考虑完整的非线性方程，周期变长
    - **影响周期修正**: $T = T_0(1 + \frac{\theta_0^2}{16} + ...)$，初始角度增加，实际周期增加。
    - **影响最大位移**: 初始角度决定了摆球的最大位移和最大势能。
    
    **6. 模拟时长**
    - **影响周期测量精度**: 时长增加，可观测更多完整周期，统计平均更准确。
    - **影响阻尼效应观测**: 较长时间可观察到完整的振幅衰减过程。
    
    这些参数关系基于精确的物理模型和非线性微分方程求解，考虑了大角度摆动的非线性效应和阻尼效应。
    """)

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
def create_pendulum_frame(x_pos, y_pos, time_val, length, gravity, frame_idx, angle_data=None, angular_velocity=None):
    """创建单摆运动的单一帧，带有详细的物理信息显示"""
    # 获取当前位置和角度
    x = x_pos[frame_idx]
    y = y_pos[frame_idx]
    
    # 如果提供了角度和角速度数据，则使用它们
    if angle_data is not None and angular_velocity is not None:
        angle = angle_data[frame_idx]
        omega = angular_velocity[frame_idx]
    else:
        # 从位置推算角度
        angle = np.arctan2(x, -y)
        omega = 0  # 无法从单帧准确推算角速度
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5*length, 1.5*length)
    ax.set_ylim(-1.5*length, 0.5*length)
    ax.grid(True)
    
    # 绘制完整轨迹线（变浅，作为背景）
    ax.plot(x_pos, y_pos, 'r-', alpha=0.1, linewidth=1)
    
    # 绘制当前轨迹
    trail_start = max(0, frame_idx - 20)  # 只显示最近的20个点
    ax.plot(x_pos[trail_start:frame_idx+1], y_pos[trail_start:frame_idx+1], 'r-', alpha=0.5, linewidth=1.5)
    
    # 绘制坐标轴
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # 绘制摆点和摆杆
    ax.plot([0], [0], 'ko', markersize=8)  # 摆点
    ax.plot([0, x], [0, y], 'k-', linewidth=2)  # 摆杆
    ax.plot([x], [y], 'ro', markersize=10)  # 摆球
    
    # 计算切向速度向量和法向加速度
    if omega != 0:
        # 切向速度
        v_scale = 0.3 * length  # 缩放因子，使箭头可见
        vx = -v_scale * omega * np.sin(angle)
        vy = -v_scale * omega * np.cos(angle)
        ax.arrow(x, y, vx, vy, head_width=0.05*length, head_length=0.08*length, 
                fc='blue', ec='blue', width=0.01*length)
        
        # 法向加速度（向心加速度）
        a_scale = 0.2 * length  # 缩放因子
        anx = -a_scale * omega**2 * np.cos(angle)
        any = a_scale * omega**2 * np.sin(angle)
        ax.arrow(x, y, anx, any, head_width=0.05*length, head_length=0.08*length, 
                fc='green', ec='green', width=0.01*length)
        
        # 切向加速度（由重力产生）
        at_scale = 0.2 * length
        atx = at_scale * gravity/length * np.sin(angle) * np.cos(angle)
        aty = at_scale * gravity/length * np.sin(angle) * np.sin(angle)
        ax.arrow(x, y, atx, aty, head_width=0.05*length, head_length=0.08*length, 
                fc='purple', ec='purple', width=0.01*length)
    
    # 添加重力示意
    grav_len = 0.2 * length
    ax.arrow(x, y, 0, grav_len, head_width=0.05*length, head_length=0.08*length, 
            fc='black', ec='black', width=0.01*length, alpha=0.5)
    ax.text(x + 0.05*length, y + grav_len/2, 'g', fontsize=12)
    
    # 绘制角度弧
    arc_radius = 0.2 * length
    theta1 = 90  # 垂直向下的角度
    theta2 = np.rad2deg(angle)
    ax.add_patch(plt.matplotlib.patches.Arc((0, 0), arc_radius*2, arc_radius*2, 
                                           theta1=theta1, theta2=theta2, color='blue'))
    ax.text(arc_radius/2 * np.sin(angle/2), -arc_radius/2 * np.cos(angle/2), 
           f'θ={np.rad2deg(angle):.1f}°', fontsize=10)
    
    # 添加文本信息
    ax.set_title('单摆运动详细分析', fontsize=14)
    
    # 添加状态信息文本框
    info_text = (
        f"时间: {time_val[frame_idx]:.2f} s\n"
        f"角度: {angle:.2f} rad ({np.rad2deg(angle):.1f}°)\n"
        f"周期: {2*np.pi*np.sqrt(length/gravity):.4f} s\n"
        f"位置: ({x:.3f}, {y:.3f}) m"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # 添加图例
    ax.legend(['轨迹', '近期轨迹', '摆点', '摆杆', '摆球'], 
             loc='lower right', fontsize=8)
    
    fig.tight_layout()
    return fig

# 创建Plotly交互式动画
def create_plotly_pendulum_animation(x_pos, y_pos, time_data, length, gravity):
    """创建Plotly交互式单摆动画"""
    # 数据准备
    pendulum_data = pd.DataFrame({
        'time': time_data,
        'x_pos': x_pos,
        'y_pos': y_pos
    })

    # 计算理论周期
    period = 2*np.pi*np.sqrt(length/gravity)
    
    # 创建基本图形
    fig = go.Figure()
    
    # 添加摆长
    fig.add_trace(
        go.Scatter(
            x=[0, x_pos[0]], 
            y=[0, y_pos[0]], 
            mode='lines', 
            line=dict(color='black', width=3),
            name='Pendulum Rod',
            showlegend=False
        )
    )
    
    # 添加摆球
    fig.add_trace(
        go.Scatter(
            x=[x_pos[0]], 
            y=[y_pos[0]], 
            mode='markers', 
            marker=dict(color='red', size=15),
            name='Pendulum Ball',
            showlegend=False
        )
    )
    
    # 添加摆点
    fig.add_trace(
        go.Scatter(
            x=[0], 
            y=[0], 
            mode='markers', 
            marker=dict(color='black', size=10),
            name='Pivot Point',
            showlegend=False
        )
    )
    
    # 添加轨迹
    fig.add_trace(
        go.Scatter(
            x=x_pos[:1], 
            y=y_pos[:1], 
            mode='lines', 
            line=dict(color='rgba(255,0,0,0.3)', width=2),
            name='Trajectory',
            showlegend=False
        )
    )
    
    # 设置布局
    axis_range = 1.5 * length
    fig.update_layout(
        title=f"Pendulum Motion (Period: {period:.4f}s)",
        xaxis=dict(
            range=[-axis_range, axis_range],
            title="X Position (m)",
            zeroline=True
        ),
        yaxis=dict(
            range=[-axis_range, 0.5 * length],
            title="Y Position (m)",
            zeroline=True,
            scaleanchor="x",  # 保持坐标轴比例一致
            scaleratio=1
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ],
            'direction': 'left',
            'pad': {'l': 10, 'r': 10, 't': 10, 'b': 10},
            'showactive': False,
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }],
        sliders=[{
            'steps': [
                {
                    'method': 'animate',
                    'label': f'{time_data[i]:.1f}s',
                    'args': [[f'frame{i}'], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                } for i in range(0, len(time_data), max(1, len(time_data) // 10))
            ],
            'x': 0.1,
            'y': 0,
            'len': 0.9,
            'pad': {'l': 10, 'r': 10, 't': 50, 'b': 10},
            'currentvalue': {
                'visible': True,
                'prefix': 'Time: ',
                'xanchor': 'right',
                'font': {'size': 12, 'color': '#666'}
            },
            'transition': {'duration': 0},
            'active': 0
        }],
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        grid=dict(rows=1, columns=1),
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(245,245,245,0.9)'
    )
    
    # 创建动画帧
    frames = []
    num_frames = min(100, len(time_data))  # 限制帧数以保持性能
    frame_indices = np.linspace(0, len(time_data) - 1, num_frames).astype(int)
    
    for i, idx in enumerate(frame_indices):
        trail_start = max(0, idx - 30)  # 只显示最近的30个点
        
        frame = go.Frame(
            name=f'frame{idx}',
            data=[
                # 更新摆杆位置
                go.Scatter(
                    x=[0, x_pos[idx]], 
                    y=[0, y_pos[idx]]
                ),
                # 更新摆球位置
                go.Scatter(
                    x=[x_pos[idx]], 
                    y=[y_pos[idx]]
                ),
                # 保持摆点不变
                go.Scatter(
                    x=[0], 
                    y=[0]
                ),
                # 更新轨迹
                go.Scatter(
                    x=x_pos[trail_start:idx+1], 
                    y=y_pos[trail_start:idx+1]
                )
            ]
        )
        frames.append(frame)
    
    fig.frames = frames
    
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
        "angular_velocity": results["angular_velocity"].tolist(),
        "x_position": results["x_position"].tolist(),
        "y_position": results["y_position"].tolist(),
        "kinetic_energy": results["kinetic_energy"].tolist(),  # 添加动能数据
        "potential_energy": results["potential_energy"].tolist(),  # 添加势能数据
        "total_energy": results["total_energy"].tolist(),
        "periods": periods,
        "avg_period": avg_period,
        "g_calculated": g_calculated
    }

if app_mode == "单摆基本模拟":
    st.header("单摆基本模拟")
    
    # 添加物理原理解释
    with st.expander("单摆物理模型详解", expanded=False):
        st.markdown("""
        ### 单摆物理模型的精确描述
        
        单摆是物理学中的一个基础模型，它由一个质点（摆球）通过一根不可伸长的轻质绳索（摆杆）连接到固定点上构成。

        #### 基本微分方程
        
        单摆运动满足如下非线性微分方程：

        $$\\frac{d^2\\theta}{dt^2} + \\frac{b}{mL}\\frac{d\\theta}{dt} + \\frac{g}{L}\\sin\\theta = 0$$

        其中：
        - $\\theta$ 是摆的偏角（从垂直方向测量）
        - $L$ 是摆长
        - $g$ 是重力加速度
        - $m$ 是摆球质量
        - $b$ 是阻尼系数（与空气阻力相关）
        
        #### 小角度近似与非线性效应
        
        当初始角度很小时（通常小于10°），可以使用小角近似 $\\sin\\theta \\approx \\theta$，此时方程简化为：

        $$\\frac{d^2\\theta}{dt^2} + \\frac{b}{mL}\\frac{d\\theta}{dt} + \\frac{g}{L}\\theta = 0$$
        
        这是一个简谐振动的方程，具有简单的解析解。然而，当角度较大时，必须考虑完整的非线性方程。

        #### 周期公式
        
        - **小角近似下的周期**：$T = 2\\pi\\sqrt{\\frac{L}{g}}$
        
        - **考虑非线性效应的周期**：$T = 2\\pi\\sqrt{\\frac{L}{g}}\\left(1 + \\frac{1}{16}\\sin^2\\frac{\\theta_0}{2} + \\frac{11}{3072}\\sin^4\\frac{\\theta_0}{2} + ...\\right)$
        
        - **考虑阻尼的周期**：$T = \\frac{2\\pi}{\\omega_0\\sqrt{1-\\zeta^2}}$，其中$\\omega_0 = \\sqrt{\\frac{g}{L}}$，$\\zeta = \\frac{b}{2m\\omega_0L}$
        
        #### 能量分析
        
        单摆的总能量由动能和势能组成：

        - **动能**：$K = \\frac{1}{2}mL^2\\left(\\frac{d\\theta}{dt}\\right)^2$
        
        - **势能**：$U = mgL(1-\\cos\\theta)$ (相对于最低点)
        
        - **总能量**：$E = K + U$
        
        在有阻尼的情况下，能量随时间损耗：$\\frac{dE}{dt} = -bL^2\\left(\\frac{d\\theta}{dt}\\right)^2$

        #### 数值求解方法
        
        本模拟使用了Runge-Kutta 4-5阶方法（RK45）求解微分方程，这是一种高精度的自适应步长积分算法，能够准确处理非线性系统的演化。相对误差容限设为 $10^{-10}$，绝对误差容限设为 $10^{-10}$，确保高精度计算。
        
        #### 重力加速度测量原理
        
        通过测量周期 $T$，可以计算重力加速度：
        
        - **基本公式**：$g = 4\\pi^2L/T^2$
        
        - **大角度修正**：$g = 4\\pi^2L/T^2 \\cdot \\left(1 - \\frac{\\sin^2(\\theta_0/2)}{16} - ...\\right)^{-2}$
        
        这种修正在初始角度大于约5.7度($0.1$ rad)时应用，可显著提高测量精度。
        """)
    
    # 使用缓存运行模拟
    with st.spinner("运行单摆模拟..."):
        sim_data = run_simulation_data(
            length, mass, gravity, damping, initial_angle_rad, t_end
        )
        
        # 从缓存数据中获取结果
        time_data = np.array(sim_data["time"])
        angle_data = np.array(sim_data["angle"])
        angular_velocity = np.array(sim_data["angular_velocity"])
        x_position = np.array(sim_data["x_position"])
        y_position = np.array(sim_data["y_position"])
        kinetic_energy = np.array(sim_data["kinetic_energy"])  # 提取动能数据
        potential_energy = np.array(sim_data["potential_energy"])  # 提取势能数据
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
        st.info("单摆运动动画。使用交互式控件播放、暂停和拖动时间滑块。")
        
        # 动画选择
        animation_type = st.radio(
            "选择动画类型",
            ["Plotly交互动画", "帧序列动画"],
            horizontal=True
        )
        
        if animation_type == "Plotly交互动画":
            # 使用Plotly创建交互式动画
            with st.spinner("生成交互式动画..."):
                plotly_fig = create_plotly_pendulum_animation(
                    x_position, y_position, time_data, length, gravity
                )
                st.plotly_chart(plotly_fig, use_container_width=True)
                st.caption("提示：使用播放按钮观看动画，拖动时间滑块可以查看特定时刻")
                
                # 添加实时数据显示面板
                st.subheader("实时物理数据")
                
                # 创建时间滑块，用于选择查看特定时刻的数据
                time_idx = st.slider(
                    "选择时间点 (秒)",
                    min_value=float(time_data[0]),
                    max_value=float(time_data[-1]),
                    value=float(time_data[0]),
                    step=float((time_data[-1] - time_data[0]) / 100),
                )
                
                # 找到最接近选定时间的索引
                closest_idx = np.argmin(np.abs(time_data - time_idx))
                
                # 创建三列布局
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 角度和角速度
                    st.metric("角度 (rad)", f"{angle_data[closest_idx]:.4f}")
                    st.metric("角速度 (rad/s)", f"{angular_velocity[closest_idx]:.4f}")
                    st.metric("位移 X (m)", f"{x_position[closest_idx]:.4f}")
                    st.metric("位移 Y (m)", f"{y_position[closest_idx]:.4f}")
                    
                with col2:
                    # 能量数据
                    st.metric("动能 (J)", f"{kinetic_energy[closest_idx]:.6f}")
                    st.metric("势能 (J)", f"{potential_energy[closest_idx]:.6f}")
                    st.metric("总能量 (J)", f"{total_energy[closest_idx]:.6f}")
                    
                    # 计算当前能量损失百分比（相对于初始能量）
                    energy_loss_percent = (1 - total_energy[closest_idx]/total_energy[0]) * 100
                    st.metric("能量损失", f"{energy_loss_percent:.2f}%", 
                             delta=f"-{energy_loss_percent:.2f}%", 
                             delta_color="inverse")
                    
                with col3:
                    # 理论与实测数据比较
                    current_period = 2 * np.pi * np.sqrt(length / gravity)
                    
                    # 修正周期（考虑大角度效应）
                    angle_correction = 1 + np.sin(initial_angle_rad/2)**2 / 16
                    corrected_period = current_period * angle_correction
                    
                    st.metric("理论周期 (s)", f"{current_period:.6f}")
                    st.metric("修正周期 (s)", f"{corrected_period:.6f}")
                    st.metric("实测周期 (s)", f"{avg_period:.6f}")
                    
                    # 计算相对误差
                    period_error = (avg_period - corrected_period) / corrected_period * 100
                    st.metric("周期相对误差", f"{period_error:.4f}%", 
                             delta=f"{period_error:.4f}%", 
                             delta_color="normal" if abs(period_error) < 1 else "inverse")
                
                # 添加动画当前时刻对应的瞬时数据可视化
                with st.expander("瞬时力学分析", expanded=True):
                    # 创建当前时刻的力学分析图
                    force_fig = plt.figure(figsize=(10, 6))
                    
                    # 1. 角度与角速度关系子图
                    ax1 = force_fig.add_subplot(1, 2, 1, polar=True)
                    # 绘制当前角度的射线
                    ax1.plot([0, angle_data[closest_idx]], [0, 1], 'r-', linewidth=2)
                    ax1.set_rticks([0.25, 0.5, 0.75, 1])
                    ax1.set_rlabel_position(angle_data[closest_idx] * 180/np.pi + 90)
                    ax1.set_title("当前角度位置")
                    
                    # 2. 能量条形图
                    ax2 = force_fig.add_subplot(1, 2, 2)
                    energy_types = ['动能', '势能', '总能量']
                    energy_values = [kinetic_energy[closest_idx], 
                                    potential_energy[closest_idx], 
                                    total_energy[closest_idx]]
                    colors = ['blue', 'red', 'green']
                    
                    ax2.bar(energy_types, energy_values, color=colors)
                    ax2.set_ylabel('能量 (J)')
                    ax2.set_title(f'能量分布 (t={time_data[closest_idx]:.2f}s)')
                    
                    # 添加初始能量水平线作为参考
                    ax2.axhline(y=total_energy[0], color='black', linestyle='--', 
                               alpha=0.7, label='初始能量')
                    ax2.legend()
                    
                    force_fig.tight_layout()
                    st.pyplot(force_fig)
                    plt.close(force_fig)
        else:
            # 原始的帧序列动画
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
            
            # 添加实时数据显示容器
            data_display_container = st.container()
            
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
                        x_position, y_position, time_data, length, gravity, idx, angle_data, angular_velocity
                    )
                    animation_container.pyplot(fig)
                    plt.close(fig)  # 关闭图表以释放内存
                    
                    # 更新实时数据显示
                    with data_display_container:
                        # 创建三列布局
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # 位置和速度数据
                            st.metric("时间 (s)", f"{time_data[idx]:.3f}")
                            st.metric("角度 (rad)", f"{angle_data[idx]:.4f}")
                            st.metric("角速度 (rad/s)", f"{angular_velocity[idx]:.4f}")
                        
                        with col2:
                            # 能量数据
                            st.metric("动能 (J)", f"{kinetic_energy[idx]:.6f}")
                            st.metric("势能 (J)", f"{potential_energy[idx]:.6f}")
                            st.metric("总能量 (J)", f"{total_energy[idx]:.6f}")
                        
                        with col3:
                            # 位置数据
                            st.metric("X位置 (m)", f"{x_position[idx]:.4f}")
                            st.metric("Y位置 (m)", f"{y_position[idx]:.4f}")
                            
                            # 计算速度大小
                            vx = length * angular_velocity[idx] * np.cos(angle_data[idx])
                            vy = length * angular_velocity[idx] * np.sin(angle_data[idx])
                            v = np.sqrt(vx**2 + vy**2)
                            st.metric("速度大小 (m/s)", f"{v:.4f}")
                    
                    # 控制帧速率
                    delay = 0.2 / frame_speed  # 基本延迟200ms，除以速度
                    time.sleep(delay)
                
                # 动画完成
                status_container.success("动画播放完成！点击'播放动画'按钮重新播放。")
            else:
                # 初始帧
                fig = create_pendulum_frame(
                    x_position, y_position, time_data, length, gravity, 0, angle_data, angular_velocity
                )
                animation_container.pyplot(fig)
                plt.close(fig)
                status_container.info("点击'播放动画'按钮开始播放。")
                
                # 显示初始状态数据
                with data_display_container:
                    st.subheader("初始状态数据")
                    
                    # 创建三列布局
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 位置和速度数据
                        st.metric("时间 (s)", f"{time_data[0]:.3f}")
                        st.metric("角度 (rad)", f"{angle_data[0]:.4f}")
                        st.metric("角速度 (rad/s)", f"{angular_velocity[0]:.4f}")
                    
                    with col2:
                        # 能量数据
                        st.metric("动能 (J)", f"{kinetic_energy[0]:.6f}")
                        st.metric("势能 (J)", f"{potential_energy[0]:.6f}")
                        st.metric("总能量 (J)", f"{total_energy[0]:.6f}")
                    
                    with col3:
                        # 位置数据
                        st.metric("X位置 (m)", f"{x_position[0]:.4f}")
                        st.metric("Y位置 (m)", f"{y_position[0]:.4f}")
                        
                        # 计算速度大小
                        vx = length * angular_velocity[0] * np.cos(angle_data[0])
                        vy = length * angular_velocity[0] * np.sin(angle_data[0])
                        v = np.sqrt(vx**2 + vy**2)
                        st.metric("速度大小 (m/s)", f"{v:.4f}")
                        
                    # 添加理论数据对比图表
                    st.subheader("理论与实测周期对比")
                    theory_period = 2 * np.pi * np.sqrt(length / gravity)
                    angle_correction = 1 + np.sin(initial_angle_rad/2)**2 / 16
                    corrected_period = theory_period * angle_correction
                    
                    data = {
                        '类型': ['小角近似周期', '大角修正周期', '实测平均周期'],
                        '周期值 (s)': [theory_period, corrected_period, avg_period]
                    }
                    
                    chart = plt.figure(figsize=(10, 4))
                    ax = chart.add_subplot(111)
                    bars = ax.bar(data['类型'], data['周期值 (s)'], color=['blue', 'green', 'red'])
                    
                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.6f}s', ha='center', va='bottom')
                    
                    ax.set_ylabel('周期 (秒)')
                    ax.set_title('周期对比')
                    chart.tight_layout()
                    
                    st.pyplot(chart)
                    plt.close(chart)
    
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
                "angular_velocity": angular_velocity,
                "x_position": x_position,
                "y_position": y_position,
                "kinetic_energy": kinetic_energy,
                "potential_energy": potential_energy,
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

st.sidebar.info("© 2025 单摆精确测量虚拟平台") 