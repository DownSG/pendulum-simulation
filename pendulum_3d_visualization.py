import sys
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, QTabWidget,
                             QDoubleSpinBox, QGroupBox, QFormLayout, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer

class MplCanvas(FigureCanvas):
    """Widget component for displaying Matplotlib charts"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)

class Pendulum3DVisualization(QMainWindow):
    """3D visualization application for pendulum motion"""
    def __init__(self):
        super().__init__()
        
        # Window settings
        self.setWindowTitle("单摆3D可视化")
        self.resize(1200, 800)
        
        # Create pendulum simulation instance
        self.pendulum = PendulumSimulation(
            length=1.0,
            mass=0.1,
            gravity=9.8,
            damping=0.1,
            initial_angle=np.pi/4
        )
        
        # Initialize parameters
        self.time_points = np.array([])
        self.angle_points = np.array([])
        self.velocity_points = np.array([])
        self.x_points = np.array([])
        self.y_points = np.array([])
        self.z_points = np.array([])
        
        self.current_index = 0
        self.is_running = False
        
        # Initialize UI
        self.init_ui()
        
        # Run initial simulation
        self.run_simulation()
        
    def init_ui(self):
        """Initialize user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for 3D view and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create 3D view widget
        self.view3d = gl.GLViewWidget()
        self.view3d.setCameraPosition(distance=4)
        
        # Create coordinate grid
        grid = gl.GLGridItem()
        grid.setSize(x=10, y=10, z=10)
        grid.setSpacing(x=0.5, y=0.5, z=0.5)
        self.view3d.addItem(grid)
        
        # Create pendulum components
        self.pendulum_line = gl.GLLinePlotItem(color=(1, 1, 1, 1), width=2)
        self.pendulum_bob = gl.GLScatterPlotItem(color=(1, 0, 0, 1), size=10)
        self.pendulum_pivot = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=(0, 1, 0, 1), size=10)
        
        # Create trajectory
        self.trajectory = gl.GLLinePlotItem(color=(0, 0.5, 1, 1), width=2)
        
        # Add items to view
        self.view3d.addItem(self.pendulum_line)
        self.view3d.addItem(self.pendulum_bob)
        self.view3d.addItem(self.pendulum_pivot)
        self.view3d.addItem(self.trajectory)
        
        # Add 3D view to left layout
        left_layout.addWidget(self.view3d, stretch=8)
        
        # Create animation control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        
        # Create play/pause button
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_animation)
        
        # Create reset button
        reset_button = QPushButton("重置")
        reset_button.clicked.connect(self.reset_animation)
        
        # Create animation speed control
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(30)
        self.speed_slider.valueChanged.connect(self.update_animation_speed)
        
        # Add widgets to control layout
        control_layout.addWidget(QLabel("速度:"))
        control_layout.addWidget(self.speed_slider, stretch=3)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(reset_button)
        
        # Add control panel to left layout
        left_layout.addWidget(control_panel, stretch=1)
        
        # Add left panel to main layout
        main_layout.addWidget(left_panel, stretch=2)
        
        # Create right panel with tabs for parameter controls and data visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Create parameters tab
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        # Create parameter controls
        param_group = QGroupBox("模拟参数")
        param_form = QFormLayout(param_group)
        
        # Length control
        self.length_spinbox = QDoubleSpinBox()
        self.length_spinbox.setRange(0.1, 5.0)
        self.length_spinbox.setValue(1.0)
        self.length_spinbox.setSingleStep(0.1)
        param_form.addRow("摆长 (m):", self.length_spinbox)
        
        # Mass control
        self.mass_spinbox = QDoubleSpinBox()
        self.mass_spinbox.setRange(0.01, 2.0)
        self.mass_spinbox.setValue(0.1)
        self.mass_spinbox.setSingleStep(0.01)
        param_form.addRow("质量 (kg):", self.mass_spinbox)
        
        # Gravity control
        self.gravity_spinbox = QDoubleSpinBox()
        self.gravity_spinbox.setRange(1.0, 20.0)
        self.gravity_spinbox.setValue(9.8)
        self.gravity_spinbox.setSingleStep(0.1)
        param_form.addRow("重力加速度 (m/s²):", self.gravity_spinbox)
        
        # Damping control
        self.damping_spinbox = QDoubleSpinBox()
        self.damping_spinbox.setRange(0.0, 1.0)
        self.damping_spinbox.setValue(0.1)
        self.damping_spinbox.setSingleStep(0.01)
        param_form.addRow("阻尼系数:", self.damping_spinbox)
        
        # Initial angle control
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0, 180)
        self.angle_spinbox.setValue(45)  # 45 degrees = pi/4 radians
        self.angle_spinbox.setSingleStep(1)
        param_form.addRow("初始角度 (°):", self.angle_spinbox)
        
        # Duration control
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(1.0, 60.0)
        self.duration_spinbox.setValue(10.0)
        self.duration_spinbox.setSingleStep(1.0)
        param_form.addRow("模拟时长 (s):", self.duration_spinbox)
        
        # Apply button
        apply_button = QPushButton("应用参数并运行模拟")
        apply_button.clicked.connect(self.run_simulation)
        
        # Add widgets to parameters layout
        params_layout.addWidget(param_group)
        params_layout.addWidget(apply_button)
        params_layout.addStretch()
        
        # Create visualization tab
        vis_tab = QWidget()
        vis_layout = QVBoxLayout(vis_tab)
        
        # Create plot selection
        plot_group = QGroupBox("图表选择")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create plot type selection dropdown
        self.plot_type = QComboBox()
        self.plot_type.addItem("角度-时间")
        self.plot_type.addItem("角速度-时间")
        self.plot_type.addItem("相空间")
        self.plot_type.addItem("能量")
        self.plot_type.currentIndexChanged.connect(self.update_plot)
        plot_layout.addWidget(self.plot_type)
        
        # Add plot group to visualization layout
        vis_layout.addWidget(plot_group)
        
        # Create matplotlib canvas for plots
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        vis_layout.addWidget(self.canvas)
        
        # Create data tab
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # Create data analysis controls
        data_group = QGroupBox("数据分析")
        data_form = QFormLayout(data_group)
        
        # Period measurement button
        period_button = QPushButton("计算周期")
        period_button.clicked.connect(self.measure_period)
        
        # Gravity calculation button
        gravity_button = QPushButton("测量重力加速度")
        gravity_button.clicked.connect(self.measure_gravity)
        
        # Add buttons to data form
        data_form.addRow("", period_button)
        data_form.addRow("", gravity_button)
        
        # Result display area
        self.result_label = QLabel("点击上面的按钮进行测量")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignTop)
        self.result_label.setStyleSheet("background-color: white; padding: 10px;")
        self.result_label.setMinimumHeight(200)
        
        # Add widgets to data layout
        data_layout.addWidget(data_group)
        data_layout.addWidget(self.result_label, stretch=1)
        
        # Add tabs
        tabs.addTab(params_tab, "参数设置")
        tabs.addTab(vis_tab, "可视化")
        tabs.addTab(data_tab, "数据分析")
        
        # Add tabs to right layout
        right_layout.addWidget(tabs)
        
        # Add right panel to main layout
        main_layout.addWidget(right_panel, stretch=1)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Create animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.update_animation_speed()
        
    def run_simulation(self):
        """Run pendulum simulation with current parameters"""
        # Stop animation if running
        if self.is_running:
            self.toggle_animation()
        
        # Get parameters from UI
        length = self.length_spinbox.value()
        mass = self.mass_spinbox.value()
        gravity = self.gravity_spinbox.value()
        damping = self.damping_spinbox.value()
        angle = np.deg2rad(self.angle_spinbox.value())  # Convert degrees to radians
        duration = self.duration_spinbox.value()
        
        # Update pendulum instance
        self.pendulum = PendulumSimulation(
            length=length,
            mass=mass,
            gravity=gravity,
            damping=damping,
            initial_angle=angle
        )
        
        # Run simulation
        self.pendulum.simulate(duration=duration, time_step=0.02)
        
        # Extract data
        self.time_points = self.pendulum.times
        self.angle_points = self.pendulum.angles
        self.velocity_points = self.pendulum.angular_velocities
        
        # Calculate 3D coordinates (using spherical coordinates)
        # x = length * sin(angle) * cos(0) = length * sin(angle) * 1 = length * sin(angle)
        # y = length * sin(angle) * sin(0) = length * sin(angle) * 0 = 0
        # z = -length * cos(angle)
        self.x_points = length * np.sin(self.angle_points)
        self.y_points = np.zeros_like(self.x_points)
        self.z_points = -length * np.cos(self.angle_points)
        
        # Reset animation
        self.reset_animation()
        
        # Update plot
        self.update_plot()
    
    def update_plot(self):
        """Update the matplotlib plot based on selected plot type"""
        # Clear previous plot
        self.canvas.axes.clear()
        
        # Get plot type
        plot_type = self.plot_type.currentText()
        
        if plot_type == "角度-时间":
            self.canvas.axes.plot(self.time_points, self.angle_points)
            self.canvas.axes.set_xlabel('时间 (s)')
            self.canvas.axes.set_ylabel('角度 (rad)')
            self.canvas.axes.set_title('角度随时间变化')
        
        elif plot_type == "角速度-时间":
            self.canvas.axes.plot(self.time_points, self.velocity_points)
            self.canvas.axes.set_xlabel('时间 (s)')
            self.canvas.axes.set_ylabel('角速度 (rad/s)')
            self.canvas.axes.set_title('角速度随时间变化')
        
        elif plot_type == "相空间":
            self.canvas.axes.plot(self.angle_points, self.velocity_points)
            self.canvas.axes.set_xlabel('角度 (rad)')
            self.canvas.axes.set_ylabel('角速度 (rad/s)')
            self.canvas.axes.set_title('相空间轨迹')
        
        elif plot_type == "能量":
            kinetic_energy, potential_energy, total_energy = self.pendulum.calculate_energy()
            
            self.canvas.axes.plot(self.time_points, kinetic_energy, 'r-', label='动能')
            self.canvas.axes.plot(self.time_points, potential_energy, 'g-', label='势能')
            self.canvas.axes.plot(self.time_points, total_energy, 'b-', label='总能量')
            
            self.canvas.axes.set_xlabel('时间 (s)')
            self.canvas.axes.set_ylabel('能量 (J)')
            self.canvas.axes.set_title('能量随时间变化')
            self.canvas.axes.legend()
        
        # Add grid and update canvas
        self.canvas.axes.grid(True)
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def toggle_animation(self):
        """Toggle animation playback"""
        if self.is_running:
            self.timer.stop()
            self.play_button.setText("播放")
        else:
            self.timer.start()
            self.play_button.setText("暂停")
        
        self.is_running = not self.is_running
    
    def reset_animation(self):
        """Reset animation to initial state"""
        # Reset animation index
        self.current_index = 0
        
        # Stop animation if running
        if self.is_running:
            self.toggle_animation()
        
        # Update 3D visualization
        self.update_pendulum_position(0)
        
        # Clear trajectory
        self.trajectory.setData(pos=np.array([[0, 0, 0]]))
    
    def update_animation_speed(self):
        """Update animation playback speed"""
        # Get speed value from slider
        speed_value = self.speed_slider.value()
        
        # Calculate interval (ms)
        interval = int(1000 / speed_value)
        
        # Update timer interval
        self.timer.setInterval(interval)
    
    def update_animation(self):
        """Update animation frame"""
        # Increment frame index
        self.current_index += 1
        
        # Check if we reached the end of animation
        if self.current_index >= len(self.time_points):
            self.current_index = 0
            self.trajectory.setData(pos=np.array([[0, 0, 0]]))
        
        # Update 3D visualization
        self.update_pendulum_position(self.current_index)
    
    def update_pendulum_position(self, index):
        """Update pendulum position in 3D view"""
        # Get coordinates at current index
        x = self.x_points[index]
        y = self.y_points[index]
        z = self.z_points[index]
        
        # Update pendulum line (from pivot to bob)
        line_points = np.array([[0, 0, 0], [x, y, z]])
        self.pendulum_line.setData(pos=line_points)
        
        # Update pendulum bob
        self.pendulum_bob.setData(pos=np.array([[x, y, z]]))
        
        # Update trajectory (keep last 100 points)
        if index > 0:
            start_idx = max(0, index - 100)
            trajectory_points = np.column_stack((
                self.x_points[start_idx:index+1],
                self.y_points[start_idx:index+1],
                self.z_points[start_idx:index+1]
            ))
            self.trajectory.setData(pos=trajectory_points)
    
    def measure_period(self):
        """Measure pendulum period"""
        # Extract angles and times
        angles = self.angle_points
        times = self.time_points
        
        # Find zero crossings (where pendulum passes through equilibrium)
        zero_crossings = np.where(np.diff(np.signbit(angles)))[0]
        
        # Calculate periods (time between every other crossing)
        periods = []
        for i in range(len(zero_crossings) - 2):
            # Only use consecutive crossings in the same direction
            if (i % 2) == 0:
                period = times[zero_crossings[i+2]] - times[zero_crossings[i]]
                periods.append(period)
        
        if periods:
            # Calculate average period and standard deviation
            avg_period = np.mean(periods)
            std_period = np.std(periods)
            
            # Calculate theoretical period (small angle approximation)
            theory_period = 2 * np.pi * np.sqrt(self.pendulum.length / self.pendulum.gravity)
            
            # Calculate error
            error = abs(avg_period - theory_period) / theory_period * 100
            
            # Update result display
            result_text = f"测量结果：\n"
            result_text += f"平均周期： {avg_period:.6f} ± {std_period:.6f} 秒\n"
            result_text += f"理论周期： {theory_period:.6f} 秒\n"
            result_text += f"相对误差： {error:.2f}%\n\n"
            result_text += f"周期测量值： {', '.join([f'{p:.6f}' for p in periods])}"
            
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("周期测量失败。尝试增加模拟时长以获得完整周期。")
    
    def measure_gravity(self):
        """Measure gravitational acceleration from period"""
        # Extract angles and times
        angles = self.angle_points
        times = self.time_points
        
        # Find zero crossings
        zero_crossings = np.where(np.diff(np.signbit(angles)))[0]
        
        # Calculate periods
        periods = []
        for i in range(len(zero_crossings) - 2):
            if (i % 2) == 0:
                period = times[zero_crossings[i+2]] - times[zero_crossings[i]]
                periods.append(period)
        
        if periods:
            # Calculate average period
            avg_period = np.mean(periods)
            
            # Calculate gravity from period (g = 4π²L/T²)
            # Apply large angle correction if needed
            if abs(self.pendulum.initial_state[0]) > 0.1:
                # Large angle correction
                correction = (1 + np.sin(self.pendulum.initial_state[0]/2)**2 / 16)**2
                g_measured = (4 * np.pi**2 * self.pendulum.length) / (avg_period**2 * correction)
            else:
                g_measured = (4 * np.pi**2 * self.pendulum.length) / (avg_period**2)
            
            # Calculate error
            error = abs(g_measured - self.pendulum.gravity) / self.pendulum.gravity * 100
            
            # Update result display
            result_text = f"重力加速度测量结果：\n"
            result_text += f"测量值： {g_measured:.6f} m/s²\n"
            result_text += f"真实值： {self.pendulum.gravity:.6f} m/s²\n"
            result_text += f"相对误差： {error:.4f}%\n\n"
            
            # Add note about angle correction
            if abs(self.pendulum.initial_state[0]) > 0.1:
                result_text += "注意: 应用了大角度修正"
            else:
                result_text += "注意: 使用了小角度近似"
            
            self.result_label.setText(result_text)
        else:
            self.result_label.setText("无法测量重力加速度。尝试增加模拟时长以获得完整周期。")

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = Pendulum3DVisualization()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 