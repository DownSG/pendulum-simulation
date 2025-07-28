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
matplotlib.use('Agg')  # 使用非交互式后端以避免字体问题
# 导入中文字体支持
try:
    from font_helper import add_chinese_font_support, setup_plotly_chinese_fonts
    # 添加中文字体支持
    add_chinese_font_support()
    # 获取Plotly字体配置
    plotly_font = setup_plotly_chinese_fonts()
    print("成功导入中文字体支持")
except Exception as e:
    print(f"导入中文字体支持失败: {e}")
    # 备用方案
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plotly_font = {'family': 'SimHei, Arial, sans-serif', 'size': 14}
import time
import pandas as pd  # 确保pandas被导入
import json

# 设置Streamlit页面配置，包括中文支持
st.set_page_config(
    page_title="单摆精密测量平台", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置CSS以确保中文字体正确显示
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .stPlotlyChart text {
        font-family: 'Noto Sans SC', sans-serif !important;
    }
    
    /* 确保Plotly图表中的中文正确显示 */
    .js-plotly-plot .plotly .main-svg text {
        font-family: 'Noto Sans SC', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("单摆精密测量平台")

st.markdown("""
本应用程序允许您模拟单摆运动，调整各种参数，并可视化结果。您可以：
- 观察单摆运动轨迹和能量变化
- 比较理论和"实验"数据
- 测量重力加速度
""")

# 创建侧边栏用于参数调整
st.sidebar.header("参数设置")

# 侧边栏中的应用选择
app_mode = st.sidebar.selectbox(
    "选择模式", 
    ["基础单摆模拟", "理论与实验对比", "重力加速度测量"]
)

# 添加参数科学说明折叠面板
with st.sidebar.expander("参数效应说明", expanded=False):
    st.markdown("""
    ### 参数对单摆运动的科学效应

    **1. Length (L)**
    - **Effect on Period**: $T = 2\pi \sqrt{L/g}$. Increasing length increases period by square root relationship.
    - **Effect on Frequency**: $f = \frac{1}{2\pi}\sqrt{g/L}$, increasing length decreases vibration frequency.
    - **Effect on Angular Acceleration**: $\ddot{\theta} = -\frac{g}{L}\sin\theta$, increasing length decreases angular acceleration.
    
    **2. Mass (m)**
    - **No Effect on Period**: In ideal pendulum, period is completely independent of mass (unlike real pendulums where angular momentum relates to mass).
    - **Effect on Damping Ratio**: $\zeta = \frac{b}{2m\omega_0L}$, increasing mass decreases damping ratio, resulting in longer vibration time.
    - **Effect on Energy Amplitude**: Increasing mass increases system total energy ($E \propto m$), but motion characteristics remain unchanged.
    
    **3. Gravity (g)**
    - **Effect on Period**: $T = 2\pi \sqrt{L/g}$. Increasing gravity decreases period, by inverse square root relationship.
    - **Effect on Restoring Force**: $F_{restore} = mg\sin\theta$. Increasing g increases restoring force, causing faster oscillation.
    - **Effect on Natural Frequency**: $\omega_0 = \sqrt{g/L}$, increasing gravity increases natural frequency.
    
    **4. Damping Coefficient (b)**
    - **Effect on Amplitude Decay Rate**: $\gamma = \frac{b}{2mL}$, increasing damping coefficient causes faster amplitude decay.
    - **Effect on Energy Loss Rate**: $\frac{dE}{dt} = -b(\frac{d\theta}{dt})^2$, increasing damping coefficient causes faster energy loss.
    - **Effect on System Type**:
        * $\zeta < 1$: Underdamped (oscillatory decay)
        * $\zeta = 1$: Critically damped (fastest return to equilibrium without oscillation)
        * $\zeta > 1$: Overdamped (slow return to equilibrium without oscillation)
    
    **5. Initial Angle ($\\theta_0$)**
    - **Effect on Nonlinearity**: Larger angles result in more significant nonlinear effects:
        * Small angles ($<10°$): $\sin\theta \approx \theta$, approximates simple harmonic motion
        * Large angles: Requires full nonlinear equation, period increases
    - **Effect on Period Correction**: $T = T_0(1 + \frac{\theta_0^2}{16} + ...)$, increasing initial angle increases actual period.
    - **Effect on Maximum Displacement**: Initial angle determines maximum bob displacement and maximum potential energy.
    
    **6. Simulation Duration**
    - **Effect on Period Measurement Precision**: Longer duration allows observing more complete periods, more accurate statistical average.
    - **Effect on Damping Observation**: Longer time allows observing complete amplitude decay process.
    
    These parameter relationships are based on precise physical models and nonlinear differential equation solutions, considering large angle nonlinear effects and damping effects.
    """)

# 通用参数设置
with st.sidebar.expander("基本参数", expanded=True):
    length = st.slider("摆长 (m)", 0.1, 2.0, 1.0, 0.1)
    mass = st.slider("质量 (kg)", 0.01, 1.0, 0.1, 0.01)
    gravity = st.slider("重力加速度 (m/s²)", 1.0, 20.0, 9.8, 0.1)
    damping = st.slider("阻尼系数", 0.0, 1.0, 0.1, 0.01)
    initial_angle = st.slider("初始角度 (°)", 5, 90, 30, 5)
    t_end = st.slider("模拟时长 (s)", 5, 30, 10, 1)

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
    
    # 确保中文字体正确设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
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
    ax.set_title('Pendulum Motion Detailed Analysis', fontsize=14)
    
    # 添加状态信息文本框
    info_text = (
        f"Time: {time_val[frame_idx]:.2f} s\n"
        f"Angle: {angle:.2f} rad ({np.rad2deg(angle):.1f}°)\n"
        f"Period: {2*np.pi*np.sqrt(length/gravity):.4f} s\n"
        f"Position: ({x:.3f}, {y:.3f}) m"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # 添加图例
    ax.legend(['Trajectory', 'Recent Trajectory', 'Pendulum Point', 'Pendulum Rod', 'Pendulum Ball'], 
             loc='lower right', fontsize=8)
    
    fig.tight_layout()
    return fig

# 创建Plotly交互式动画
def create_plotly_pendulum_animation(x_pos, y_pos, time_data, length, gravity, angle_data=None, angular_velocity=None, kinetic_energy=None, potential_energy=None):
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
    
    # 设置Plotly中文字体
    chinese_font = dict(
        family="'Noto Sans SC', 'Microsoft YaHei', 'SimHei', Arial, sans-serif",
        size=14,
        color="black"
    )
    
    # 添加摆长
    fig.add_trace(
        go.Scatter(
            x=[0, x_pos[0]], 
            y=[0, y_pos[0]], 
            mode='lines', 
            line=dict(color='black', width=3),
            name='摆杆',
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
            name='摆球',
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
            name='摆点',
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
            name='轨迹',
            showlegend=False
        )
    )
    
    # 设置布局
    axis_range = 1.5 * length
    fig.update_layout(
        title=dict(
            text=f"单摆运动动画 (周期: {period:.4f}s)",
            font=chinese_font
        ),
        xaxis=dict(
            range=[-axis_range, axis_range],
            title=dict(text="X位置 (m)", font=chinese_font),
            zeroline=True
        ),
        yaxis=dict(
            range=[-axis_range, 0.5 * length],
            title=dict(text="Y位置 (m)", font=chinese_font),
            zeroline=True,
            scaleanchor="x",  # 保持坐标轴比例一致
            scaleratio=1
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                dict(
                    label="播放",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="暂停",
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
            'yanchor': 'top',
            'font': chinese_font
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
                'prefix': '时间: ',
                'xanchor': 'right',
                'font': chinese_font
            },
            'transition': {'duration': 0},
            'active': 0
        }],
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        grid=dict(rows=1, columns=1),
        paper_bgcolor='rgba(255,255,255,0.9)',
        plot_bgcolor='rgba(245,245,245,0.9)',
        font=chinese_font,  # 设置全局字体
        uirevision=True  # 确保UI状态保持一致
    )
    
    # 创建动画帧
    frames = []
    num_frames = min(100, len(time_data))  # 限制帧数以保持性能
    frame_indices = np.linspace(0, len(time_data) - 1, num_frames).astype(int)
    
    for i, idx in enumerate(frame_indices):
        trail_start = max(0, idx - 30)  # 只显示最近的30个点
        
        # 准备当前帧的数据
        current_data = [
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
        
        # 如果提供了角度数据，添加实时数据注释
        if angle_data is not None and angular_velocity is not None and kinetic_energy is not None and potential_energy is not None:
            # 添加实时数据文本
            data_text = (
                f"时间: {time_data[idx]:.2f}s<br>"
                f"角度: {angle_data[idx]:.3f}rad<br>"
                f"角速度: {angular_velocity[idx]:.3f}rad/s<br>"
                f"动能: {kinetic_energy[idx]:.3f}J<br>"
                f"势能: {potential_energy[idx]:.3f}J<br>"
                f"总能量: {kinetic_energy[idx] + potential_energy[idx]:.3f}J"
            )
            
            # 添加数据注释
            current_data.append(
                go.Scatter(
                    x=[-axis_range * 0.9],
                    y=[-axis_range * 0.8],
                    mode='text',
                    text=data_text,
                    textposition="top right",
                    textfont=chinese_font,
                    showlegend=False
                )
            )
        
        frame = go.Frame(
            name=f'frame{idx}',
            data=current_data
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # 应用中文字体设置
    fig = ensure_plotly_chinese_fonts(fig)
    
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
    results = pendulum.simulate(duration=t_end, time_step=t_end/500)
    
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

# 确保Plotly图表中的中文显示正常
def ensure_plotly_chinese_fonts(fig):
    """确保Plotly图表中的中文字体正确显示"""
    # 设置通用的中文字体
    chinese_font = {
        'family': "'Noto Sans SC', 'Microsoft YaHei', 'SimHei', Arial, sans-serif",
        'size': 14
    }
    
    # 更新图表的字体配置
    fig.update_layout(
        font=chinese_font,
        title_font=chinese_font,
        legend_title_font=chinese_font
    )
    
    # 更新坐标轴标题字体
    fig.update_xaxes(title_font=chinese_font)
    fig.update_yaxes(title_font=chinese_font)
    
    # 更新图例字体
    fig.update_layout(legend_font=chinese_font)
    
    return fig

if app_mode == "基础单摆模拟":
    st.header("基础单摆模拟")
    
    # 添加物理模型描述
    with st.expander("单摆物理模型精确描述"):
        st.markdown("""
        单摆是物理学中的一个基础模型，由一个质点（摆锤）通过刚性、无质量的绳索（摆杆）连接到固定点组成。

        ### 基本微分方程
        单摆运动满足以下非线性微分方程：

        $$\\frac{d^2\\theta}{dt^2} + \\frac{b}{mL}\\frac{d\\theta}{dt} + \\frac{g}{L}\\sin\\theta = 0$$

        其中：
        - $\\theta$ 是单摆的角位移（从垂直方向测量）
        - $L$ 是摆长
        - $g$ 是重力加速度
        - $m$ 是摆锤质量
        - $b$ 是阻尼系数（与空气阻力相关）

        ### 小角度近似和非线性效应
        当初始角度很小时（通常小于10°），可以使用小角度近似 $\\sin\\theta \\approx \\theta$，将方程简化为：

        $$\\frac{d^2\\theta}{dt^2} + \\frac{b}{mL}\\frac{d\\theta}{dt} + \\frac{g}{L}\\theta = 0$$

        这是一个简单谐振动方程，有直接的解析解。但当角度较大时，必须考虑完整的非线性方程。

        ### 周期公式
        小角度近似下的周期：$T = 2\\pi\\sqrt{\\frac{L}{g}}$

        考虑非线性效应的周期：$T = 2\\pi\\sqrt{\\frac{L}{g}}(1 + \\frac{1}{16}\\sin^2\\frac{\\theta_0}{2} + \\frac{11}{3072}\\sin^4\\frac{\\theta_0}{2} + ...)$

        考虑阻尼的周期：$T = \\frac{2\\pi}{\\omega_0\\sqrt{1-\\zeta^2}}$，其中 $\\omega_0 = \\sqrt{\\frac{g}{L}}$，$\\zeta = \\frac{b}{2m\\omega_0L}$

        ### 能量分析
        单摆的总能量由动能和势能组成：

        动能：$K = \\frac{1}{2}mL^2(\\frac{d\\theta}{dt})^2$

        势能：$U = mgL(1-\\cos\\theta)$（相对于最低点）

        总能量：$E = K + U$

        存在阻尼时，能量随时间耗散：$\\frac{dE}{dt} = -bL^2(\\frac{d\\theta}{dt})^2$

        ### 数值解法
        本模拟使用龙格-库塔4-5阶方法（RK45）求解微分方程，这是一种高精度自适应步长积分算法，能够准确处理非线性系统的演化。相对误差容限设为 $10^{-10}$，绝对误差容限设为 $10^{-10}$，确保高精度计算。

        ### 重力加速度测量原理
        通过测量周期 $T$，可以计算重力加速度：

        基本公式：$g = 4\\pi^2L/T^2$

        大角度修正：$g = 4\\pi^2L/T^2 \\cdot (1-\\frac{\\sin^2(\\theta_0/2)}{16}-...)^{-2}$

        当初始角度大于约5.7度（$0.1$ 弧度）时，应用此修正可显著提高测量精度。
        """)
    
    # 添加参数对单摆运动的科学效应说明
    with st.expander("参数对单摆运动的科学效应"):
        st.markdown(r"""
        ## 参数对单摆运动的科学效应

        ### 1. 摆长 (L)

        - **对周期的影响**: $T = 2\pi\sqrt{\frac{L}{g}}$。增加摆长会按照平方根关系增加周期。
        - **对频率的影响**: $f = \frac{1}{2\pi}\sqrt{\frac{g}{L}}$，增加摆长会降低振动频率。
        - **对角加速度的影响**: $\ddot{\theta} = -\frac{g}{L}\sin\theta$，增加摆长会降低角加速度。

        ### 2. 质量 (m)

        - **对周期无影响**: 在理想单摆中，周期完全独立于质量（与实际单摆不同，实际单摆的角动量与质量相关）。
        - **对阻尼比的影响**: $\zeta = \frac{b}{2m\omega_0L}$，增加质量会降低阻尼比，导致振动时间更长。
        - **对能量幅度的影响**: 增加质量会增加系统总能量（$E \propto m$），但运动特性保持不变。

        ### 3. 重力加速度 (g)

        - **对周期的影响**: $T = 2\pi\sqrt{\frac{L}{g}}$。增加重力会按照反平方根关系减小周期。
        - **对恢复力的影响**: $F_{restore} = mg\sin\theta$。增加g会增加恢复力，导致更快的振荡。
        - **对自然频率的影响**: $\omega_0 = \sqrt{\frac{g}{L}}$，增加重力会增加自然频率。

        ### 4. 阻尼系数 (b)

        - **对振幅衰减率的影响**: $\gamma = \frac{b}{2mL}$，增加阻尼系数会导致振幅更快衰减。
        - **对能量损失率的影响**: $\frac{dE}{dt} = -b(\frac{d\theta}{dt})^2$，增加阻尼系数会导致能量更快损失。
        - **对系统类型的影响**:
          - $\zeta < 1$: 欠阻尼（振荡衰减）
          - $\zeta = 1$: 临界阻尼（最快回到平衡位置而无振荡）
          - $\zeta > 1$: 过阻尼（缓慢回到平衡位置而无振荡）

        ### 5. 初始角度 ($\theta_0$)

        - **对非线性的影响**: 更大的角度会导致更显著的非线性效应:
          - 小角度（$< 10°$）: $\sin\theta \approx \theta$，近似于简谐运动
          - 大角度: 需要完整的非线性方程，周期增加
        - **对周期修正的影响**: $T = T_0(1 + \frac{\theta_0^2}{16} + ...)$，增加初始角度会增加实际周期。
        - **对最大位移的影响**: 初始角度决定了摆锤的最大位移和最大势能。

        ### 6. 模拟持续时间

        - **对周期测量精度的影响**: 更长的持续时间允许观察更多完整周期，统计平均更准确。
        - **对阻尼观察的影响**: 更长的时间允许观察完整的振幅衰减过程。

        这些参数关系基于精确的物理模型和非线性微分方程解，考虑了大角度非线性效应和阻尼效应。
        """)
    
    # 使用缓存运行模拟
    with st.spinner("Running pendulum simulation..."):
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
        st.metric("测量平均周期", f"{avg_period:.6f} s")
    with col2:
        st.metric("计算得到的重力加速度", f"{g_calculated:.6f} m/s²")
    with col3:
        theory_period = 2 * np.pi * np.sqrt(length / gravity)
        error = abs(avg_period - theory_period)/theory_period*100
        st.metric("理论周期误差", f"{error:.4f}%")
    
    # 创建图表选项卡
    tab1, tab2, tab3 = st.tabs(["动画", "图表", "数据"])
    
    with tab1:
        # 动画设置
        st.info("单摆运动动画。使用交互控件播放、暂停，并拖动时间滑块。")
        
        # 动画选择
        animation_type = st.radio(
            "选择动画类型",
            ["Plotly交互式动画", "帧序列动画"],
            horizontal=True
        )
        
        if animation_type == "Plotly交互式动画":
            # 使用Plotly创建交互式动画
            with st.spinner("Generating interactive animation..."):
                plotly_fig = create_plotly_pendulum_animation(
                    x_position, y_position, time_data, length, gravity,
                    angle_data, angular_velocity, kinetic_energy, potential_energy
                )
                # 确保中文字体正确显示
                plotly_fig = ensure_plotly_chinese_fonts(plotly_fig)
                st.plotly_chart(plotly_fig, use_container_width=True)
                st.caption("提示：使用播放按钮观看动画，拖动时间滑块查看特定时刻。动画中包含实时数据。")
                
                # 添加实时数据显示面板
                with st.expander("实时物理数据", expanded=True):
                    # 时间选择滑块
                    st.subheader("选择时间点 (秒)")
                    selected_time_idx = st.slider(
                        "选择时间点",
                        min_value=0,
                        max_value=len(time_data) - 1,
                        value=0,
                        format="%d",
                        key="time_slider"
                    )
                    selected_time = time_data[selected_time_idx]
                    
                    # 显示选定时间点的数据
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("运动学数据")
                        st.write(f"时间: {selected_time:.3f} 秒")
                        angle_deg = np.degrees(angle_data[selected_time_idx])
                        st.write(f"角度: {angle_data[selected_time_idx]:.4f} 弧度 ({angle_deg:.1f}°)")
                        st.write(f"角速度: {angular_velocity[selected_time_idx]:.4f} 弧度/秒")
                        st.write(f"X位置: {x_position[selected_time_idx]:.4f} 米")
                        st.write(f"Y位置: {y_position[selected_time_idx]:.4f} 米")
                        
                        # 周期和g值估计
                        st.subheader("实时周期和重力加速度估计")
                        # 创建临时PendulumSimulation对象来计算周期
                        temp_pendulum = PendulumSimulation(length=length, mass=mass, gravity=gravity, damping=damping)
                        # 计算周期数
                        num_periods, avg_period = temp_pendulum.calculate_periods(angle_data, time_data)
                        if num_periods > 0:
                            estimated_g = 4 * np.pi**2 * length / (avg_period**2)
                            st.write(f"观测到的周期数: {num_periods} - 重力加速度估计值: {estimated_g:.4f} m/s²")
                        else:
                            st.write("观测到的周期数: 0 - 重力加速度估计值: 暂无")
                    
                    with col2:
                        st.subheader("动力学数据")
                        # 计算速度大小
                        v_x = -length * angular_velocity[selected_time_idx] * np.sin(angle_data[selected_time_idx])
                        v_y = length * angular_velocity[selected_time_idx] * np.cos(angle_data[selected_time_idx])
                        speed = np.sqrt(v_x**2 + v_y**2)
                        st.write(f"速度大小: {speed:.4f} 米/秒")
                        
                        # 计算加速度分量
                        a_tangential = -gravity * np.sin(angle_data[selected_time_idx])
                        a_normal = length * angular_velocity[selected_time_idx]**2
                        a_total = np.sqrt(a_tangential**2 + a_normal**2)
                        st.write(f"切向加速度: {a_tangential:.4f} 米/秒²")
                        st.write(f"法向加速度: {a_normal:.4f} 米/秒²")
                        st.write(f"总加速度: {a_total:.4f} 米/秒²")
                        
                        # 计算恢复力
                        restoring_force = mass * gravity * np.sin(angle_data[selected_time_idx])
                        st.write(f"恢复力: {restoring_force:.4f} 牛顿")
                        
                        # 能量数据
                        st.subheader("能量数据")
                        st.write(f"动能: {kinetic_energy[selected_time_idx]:.6f} 焦耳")
                        st.write(f"势能: {potential_energy[selected_time_idx]:.6f} 焦耳")
                        st.write(f"总能量: {kinetic_energy[selected_time_idx] + potential_energy[selected_time_idx]:.6f} 焦耳")
                        
                        # 能量损失百分比
                        initial_energy = kinetic_energy[0] + potential_energy[0]
                        current_energy = kinetic_energy[selected_time_idx] + potential_energy[selected_time_idx]
                        energy_loss_percent = ((initial_energy - current_energy) / initial_energy * 100) if initial_energy > 0 else 0
                        st.write(f"能量损失: {energy_loss_percent:.2f}%")
                        
                        # 功率损失（近似计算）
                        if selected_time_idx > 0:
                            time_diff = time_data[selected_time_idx] - time_data[selected_time_idx - 1]
                            energy_diff = (kinetic_energy[selected_time_idx - 1] + potential_energy[selected_time_idx - 1]) - current_energy
                            power_loss = energy_diff / time_diff if time_diff > 0 else 0
                            st.write(f"功率损失: {power_loss:.6f} 瓦特")
                
                # 添加瞬时力学分析可视化
                with st.expander("瞬时力学分析", expanded=True):
                    # 创建当前时刻的力学分析图
                    force_fig = plt.figure(figsize=(10, 6))
                    
                    # 确保中文字体正确设置
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 1. 角度与角速度关系子图（相位图）
                    ax1 = force_fig.add_subplot(1, 2, 1)
                    # 绘制相位轨迹
                    ax1.plot(angle_data, angular_velocity, 'b-', alpha=0.3)
                    # 标记当前点
                    ax1.plot(angle_data[selected_time_idx], angular_velocity[selected_time_idx], 'ro')
                    ax1.set_xlabel('角度 (rad)')
                    ax1.set_ylabel('角速度 (rad/s)')
                    ax1.set_title('相空间图')
                    ax1.grid(True)
                    
                    # 2. 能量条形图
                    ax2 = force_fig.add_subplot(1, 2, 2)
                    energy_types = ['动能', '势能', '总能量']
                    energy_values = [kinetic_energy[selected_time_idx], 
                                    potential_energy[selected_time_idx], 
                                    total_energy[selected_time_idx]]
                    colors = ['blue', 'red', 'green']
                    
                    bars = ax2.bar(energy_types, energy_values, color=colors)
                    
                    # 添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{height:.4f}J', ha='center', va='bottom', fontsize=9)
                    
                    ax2.set_ylabel('能量 (J)')
                    ax2.set_title(f'能量分布 (t={time_data[selected_time_idx]:.2f}s)')
                    
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
            st.subheader("实时物理数据")
            data_col1 = st.empty()
            data_col2 = st.empty()
            data_col3 = st.empty()
            
            # 创建三列布局用于实时数据显示
            data_cols = st.columns(3)
            
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
                    with data_cols[0]:
                        data_col1.markdown(f"""
                        ### 运动学数据
                        - **时间**: {time_data[idx]:.3f} s
                        - **角度**: {angle_data[idx]:.4f} rad ({np.rad2deg(angle_data[idx]):.1f}°)
                        - **角速度**: {angular_velocity[idx]:.4f} rad/s
                        - **X位置**: {x_position[idx]:.4f} m
                        - **Y位置**: {y_position[idx]:.4f} m
                        """)
                    
                    with data_cols[1]:
                        # 计算速度
                        vx = length * angular_velocity[idx] * np.cos(angle_data[idx])
                        vy = length * angular_velocity[idx] * np.sin(angle_data[idx])
                        v_mag = np.sqrt(vx**2 + vy**2)
                        
                        # 计算加速度
                        a_tan = gravity * np.sin(angle_data[idx])  # 切向加速度
                        a_n = v_mag**2 / length  # 法向加速度
                        a_mag = np.sqrt(a_tan**2 + a_n**2)  # 总加速度
                        
                        data_col2.markdown(f"""
                        ### 动力学数据
                        - **速度大小**: {v_mag:.4f} m/s
                        - **切向加速度**: {a_tan:.4f} m/s²
                        - **法向加速度**: {a_n:.4f} m/s²
                        - **总加速度**: {a_mag:.4f} m/s²
                        - **恢复力**: {mass * gravity * np.sin(angle_data[idx]):.4f} N
                        """)
                    
                    with data_cols[2]:
                        # 能量数据
                        energy_loss = (1 - total_energy[idx]/total_energy[0]) * 100
                        
                        data_col3.markdown(f"""
                        ### 能量数据
                        - **动能**: {kinetic_energy[idx]:.6f} J
                        - **势能**: {potential_energy[idx]:.6f} J
                        - **总能量**: {total_energy[idx]:.6f} J
                        - **能量损失**: {energy_loss:.2f}%
                        - **功率损失**: {damping * (angular_velocity[idx]**2):.6f} W
                        """)
                    
                    # 控制帧速率
                    delay = 0.2 / frame_speed  # 基本延迟200ms，除以速度
                    time.sleep(delay)
                
                # 动画完成
                status_container.success("Animation playback complete! Click 'Play Animation' to replay.")
            else:
                # 初始帧
                fig = create_pendulum_frame(
                    x_position, y_position, time_data, length, gravity, 0, angle_data, angular_velocity
                )
                animation_container.pyplot(fig)
                plt.close(fig)
                status_container.info("Click 'Play Animation' to start playback.")
                
                # 显示初始状态数据
                with data_cols[0]:
                    data_col1.markdown(f"""
                    ### 初始运动学数据
                    - **时间**: {time_data[0]:.3f} s
                    - **角度**: {angle_data[0]:.4f} rad ({np.rad2deg(angle_data[0]):.1f}°)
                    - **角速度**: {angular_velocity[0]:.4f} rad/s
                    - **X位置**: {x_position[0]:.4f} m
                    - **Y位置**: {y_position[0]:.4f} m
                    """)
                
                with data_cols[1]:
                    # 计算初始速度
                    vx = length * angular_velocity[0] * np.cos(angle_data[0])
                    vy = length * angular_velocity[0] * np.sin(angle_data[0])
                    v_mag = np.sqrt(vx**2 + vy**2)
                    
                    # 计算初始加速度
                    a_tan = gravity * np.sin(angle_data[0])  # 切向加速度
                    a_n = v_mag**2 / length  # 法向加速度
                    a_mag = np.sqrt(a_tan**2 + a_n**2)  # 总加速度
                    
                    data_col2.markdown(f"""
                    ### 初始动力学数据
                    - **速度大小**: {v_mag:.4f} m/s
                    - **切向加速度**: {a_tan:.4f} m/s²
                    - **法向加速度**: {a_n:.4f} m/s²
                    - **总加速度**: {a_mag:.4f} m/s²
                    - **恢复力**: {mass * gravity * np.sin(angle_data[0]):.4f} N
                    """)
                
                with data_cols[2]:
                    data_col3.markdown(f"""
                    ### 初始能量数据
                    - **动能**: {kinetic_energy[0]:.6f} J
                    - **势能**: {potential_energy[0]:.6f} J
                    - **总能量**: {total_energy[0]:.6f} J
                    - **能量损失**: 0.00%
                    - **功率损失**: {damping * (angular_velocity[0]**2):.6f} W
                    """)
                
                # 添加理论数据对比图表
                st.subheader("理论与测量周期对比")
                theory_period = 2 * np.pi * np.sqrt(length / gravity)
                angle_correction = 1 + np.sin(initial_angle_rad/2)**2 / 16
                corrected_period = theory_period * angle_correction
                
                data = {
                    'Type': ['小角度周期', '大角度修正周期', '测量平均周期'],
                    'Period Value (s)': [theory_period, corrected_period, avg_period]
                }
                
                chart = plt.figure(figsize=(10, 4))
                
                # 确保中文字体正确设置
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                
                ax = chart.add_subplot(111)
                bars = ax.bar(data['Type'], data['Period Value (s)'], color=['blue', 'green', 'red'])
                
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
        with st.spinner("Generating charts..."):
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
            
            # 设置模拟结果
            pendulum.times = time_data
            pendulum.angles = angle_data
            pendulum.angular_velocities = angular_velocity
            
            # 生成图表
            fig = pendulum.visualize()
            # 不再需要应用英文标签，因为我们已经改为中文
            st.pyplot(fig)
    
    with tab3:
        # 显示数据
        st.subheader("Simulation Data")
        data = {
            "Time (s)": time_data,
            "Angle (rad)": angle_data,
            "X Position (m)": x_position,
            "Y Position (m)": y_position,
            "Total Energy (J)": total_energy
        }
        
        # 仅显示部分数据点
        display_points = min(100, len(time_data))
        step = len(time_data) // display_points
        
        for key in data:
            data[key] = data[key][::step]
            
        st.dataframe(data)
        
        # 添加下载按钮
        csv = np.column_stack([data["Time (s)"], data["Angle (rad)"], data["X Position (m)"], data["Y Position (m)"], data["Total Energy (J)"]])
        
        # 使用pandas导出为CSV，以处理中文编码问题
        import pandas as pd
        csv_df = pd.DataFrame({
            'time': data["Time (s)"], 
            'angle': data["Angle (rad)"], 
            'x_pos': data["X Position (m)"], 
            'y_pos': data["Y Position (m)"], 
            'energy': data["Total Energy (J)"]
        })
        
        # 转换为CSV字符串
        csv_buffer = BytesIO()
        csv_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        
        st.download_button(
            label="Download Data as CSV",
            data=csv_buffer,
            file_name="pendulum_simulation_data.csv",
            mime="text/csv"
        )

elif app_mode == "理论与实验对比":
    st.header("理论与实验对比")
    
    # 添加额外的实验参数
    with st.sidebar.expander("实验参数", expanded=True):
        exp_gravity = st.slider("实验重力加速度 (m/s²)", 9.5, 10.0, 9.81, 0.01)
        exp_damping = st.slider("实验阻尼系数", 0.05, 0.5, 0.15, 0.01)
        exp_angle_error = st.slider("初始角度误差 (°)", -5.0, 5.0, 1.0, 0.5)
        noise_level = st.slider("噪声水平", 0.001, 0.02, 0.005, 0.001)
    
    # 转换为弧度
    exp_angle_error_rad = np.deg2rad(exp_angle_error)
    
    # 创建理论模型
    with st.spinner("正在运行理论模型..."):
        theory = PendulumSimulation(
            length=length,
            mass=mass,
            gravity=gravity,
            damping=0.01,  # 极小阻尼
            initial_angle=initial_angle_rad
        )
        theory_results = theory.simulate(duration=t_end, time_step=t_end/500)
    
    # 创建实验模型
    with st.spinner("正在运行实验模型..."):
        experiment = PendulumSimulation(
            length=length,
            mass=mass,
            gravity=exp_gravity,
            damping=exp_damping,
            initial_angle=initial_angle_rad + exp_angle_error_rad
        )
        exp_results = experiment.simulate(duration=t_end, time_step=t_end/500)
    
    # 添加噪声
    with st.spinner("正在添加测量噪声..."):
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
    with st.spinner("正在计算误差指标..."):
        error_metrics = analyzer.calculate_error_metrics()
    
    # 显示误差指标
    st.subheader("误差指标")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("均方根误差 (RMSE)", f"{error_metrics['RMSE']:.6f}")
    with col2:
        st.metric("平均绝对误差 (MAE)", f"{error_metrics['MAE']:.6f}")
    with col3:
        st.metric("平均相对误差", f"{error_metrics['MAPE']:.2f}%")
    
    # 可视化比较
    st.subheader("数据对比可视化")
    with st.spinner("正在生成对比图表..."):
        fig = analyzer.visualize_comparison()
        st.pyplot(fig)
    
    # 保存数据
    if st.button("保存对比数据为CSV"):
        analyzer.export_data_to_csv('pendulum_data_comparison.csv')
        st.success("数据已保存至 pendulum_data_comparison.csv")

elif app_mode == "重力加速度测量":
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
    with st.spinner("正在为不同摆长测量周期..."):
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
    
    # 不再需要修改图表标签，因为DataAnalyzer中已经使用中文标签
    
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
    ### 如何使用本应用程序
    1. 在左侧边栏选择模拟模式
    2. 调整各种参数以观察它们的影响
    3. 查看图表、数据和动画以理解单摆运动规律
    
    ### 参数解释
    - **摆长**：从支点到摆球中心的距离
    - **质量**：摆球质量，影响能量但不影响周期
    - **重力加速度**：影响单摆周期的主要参数
    - **阻尼系数**：模拟空气阻力等阻尼效应
    - **初始角度**：释放单摆的初始角度
    
    ### 实验背景
    单摆是经典物理学中研究周期运动和测量重力加速度的重要实验。通过测量不同摆长下的周期，
    可以使用公式 T = 2π√(L/g) 精确计算重力加速度。
    """)

st.sidebar.info("© 2025 单摆精密测量虚拟平台") 

# 在侧边栏增加本地分析结果加载
st.sidebar.header("数据加载与显示")
load_mode = st.sidebar.radio("选择数据来源", ["在线模拟", "加载本地分析结果"])

if load_mode == "加载本地分析结果":
    st.header("本地分析结果显示")
    uploaded_file = st.sidebar.file_uploader("上传本地JSON分析结果", type=["json"])
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                import pandas as pd
                df = pd.DataFrame(data)
                st.subheader("数据表预览")
                st.dataframe(df)
                # 数据下载
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("下载数据为CSV", csv, file_name="analysis_result.csv", mime="text/csv")
                # 多列联动可视化
                st.subheader("多列对比可视化")
                num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                selected_cols = st.multiselect("选择要比较的数值字段", num_cols, default=num_cols[:2])
                chart_html = ""
                if selected_cols:
                    st.line_chart(df[selected_cols])
                    # 生成图表HTML（静态图片）
                    import matplotlib.pyplot as plt
                    import io, base64
                    fig, ax = plt.subplots()
                    df[selected_cols].plot(ax=ax)
                    ax.set_title("Selected Columns Comparison")
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    chart_html = f'<img src="data:image/png;base64,{img_base64}"/>'
                # 生成实验报告
                st.subheader("自动生成实验报告")
                with st.form("report_form"):
                    report_title = st.text_input("报告标题", "单摆实验分析报告")
                    report_author = st.text_input("作者", "")
                    report_desc = st.text_area("实验描述", "本报告基于本地分析结果自动生成，包括数据表格、主要图表和结论。")
                    submitted = st.form_submit_button("生成HTML报告")
                if submitted:
                    html = f"""
                    <html><head><meta charset='utf-8'><title>{report_title}</title></head><body>
                    <h1>{report_title}</h1>
                    <p><b>作者：</b>{report_author}</p>
                    <p><b>实验描述：</b>{report_desc}</p>
                    <h2>数据表</h2>
                    {df.to_html(index=False)}
                    <h2>主要图表</h2>
                    {chart_html}
                    <h2>主要结论</h2>
                    <ul>
                    {''.join(f'<li>{col}：最大值={df[col].max():.4g}，最小值={df[col].min():.4g}，平均值={df[col].mean():.4g}</li>' for col in selected_cols)}
                    </ul>
                    </body></html>
                    """
                    st.download_button("下载HTML实验报告", html, file_name="pendulum_report.html", mime="text/html")
            elif isinstance(data, dict):
                st.json(data)
                st.info("已加载单个分析结果（如g值测量、阻尼分析等）")
            else:
                st.warning("Unrecognized JSON data structure!")
        except Exception as e:
            st.error(f"JSON parsing failed: {e}")
    st.stop() 