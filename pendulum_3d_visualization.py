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

from pendulum_simulation import PendulumSimulation
from data_analyzer import DataAnalyzer

class MplCanvas(FigureCanvas):
    """用于显示Matplotlib图表的QWidget组件"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class Pendulum3DWidget(gl.GLViewWidget):
    """单摆的3D可视化部件"""
    def __init__(self, parent=None):
        super(Pendulum3DWidget, self).__init__(parent)
        # 设置相机位置
        self.setCameraPosition(distance=10, azimuth=-65, elevation=20)
        
        # 添加坐标轴
        self.addItem(gl.GLAxisItem(size=pg.Vector(5, 5, 5)))
        
        # 添加网格
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        grid.setSpacing(1, 1, 0)
        self.addItem(grid)
        
        # 初始化摆杆和摆球
        self.pendulum_rod = gl.GLLinePlotItem(color=(255, 255, 255, 255), width=3)
        self.addItem(self.pendulum_rod)
        
        # 创建摆球（球体）
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.2)
        self.pendulum_bob = gl.GLMeshItem(meshdata=md, smooth=True, color=(1.0, 0.0, 0.0, 1.0))
        self.addItem(self.pendulum_bob)
        
        # 创建支点（小球体）
        md = gl.MeshData.sphere(rows=10, cols=20, radius=0.1)
        self.pivot = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.5, 0.5, 0.5, 1.0))
        self.pivot.translate(0, 0, 0)
        self.addItem(self.pivot)
        
        # 轨迹存储
        self.trajectory_points = []
        self.trajectory = gl.GLLinePlotItem(color=(255, 255, 0, 100), width=2)
        self.addItem(self.trajectory)
        
    def update_pendulum(self, x, y, z, length):
        """更新单摆位置"""
        # 更新摆杆
        self.pendulum_rod.setData(pos=np.array([[0, 0, 0], [x, y, z]]))
        
        # 更新摆球位置
        self.pendulum_bob.resetTransform()
        self.pendulum_bob.translate(x, y, z)
        
        # 更新轨迹
        self.trajectory_points.append([x, y, z])
        if len(self.trajectory_points) > 200:  # 限制轨迹点数量
            self.trajectory_points = self.trajectory_points[-200:]
        self.trajectory.setData(pos=np.array(self.trajectory_points))
        
    def clear_trajectory(self):
        """清除轨迹"""
        self.trajectory_points = []
        self.trajectory.setData(pos=np.array([]))

class PendulumControlPanel(QWidget):
    """单摆控制面板"""
    parameter_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super(PendulumControlPanel, self).__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        # 创建表单布局
        form_layout = QFormLayout()
        
        # 摆长控制
        self.length_spinbox = QDoubleSpinBox()
        self.length_spinbox.setRange(0.1, 5.0)
        self.length_spinbox.setValue(1.0)
        self.length_spinbox.setSingleStep(0.1)
        self.length_spinbox.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("摆长 (m):", self.length_spinbox)
        
        # 质量控制
        self.mass_spinbox = QDoubleSpinBox()
        self.mass_spinbox.setRange(0.01, 2.0)
        self.mass_spinbox.setValue(0.1)
        self.mass_spinbox.setSingleStep(0.01)
        form_layout.addRow("质量 (kg):", self.mass_spinbox)
        
        # 重力加速度控制
        self.gravity_spinbox = QDoubleSpinBox()
        self.gravity_spinbox.setRange(1.0, 20.0)
        self.gravity_spinbox.setValue(9.8)
        self.gravity_spinbox.setSingleStep(0.1)
        self.gravity_spinbox.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("重力加速度 (m/s²):", self.gravity_spinbox)
        
        # 阻尼系数控制
        self.damping_spinbox = QDoubleSpinBox()
        self.damping_spinbox.setRange(0.0, 1.0)
        self.damping_spinbox.setValue(0.1)
        self.damping_spinbox.setSingleStep(0.01)
        form_layout.addRow("阻尼系数:", self.damping_spinbox)
        
        # 初始角度控制
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(0, 90)
        self.angle_spinbox.setValue(30)
        self.angle_spinbox.setSingleStep(5)
        self.angle_spinbox.valueChanged.connect(self.on_parameter_changed)
        form_layout.addRow("初始角度 (°):", self.angle_spinbox)
        
        # 模拟时长控制
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setRange(5, 30)
        self.duration_spinbox.setValue(10)
        self.duration_spinbox.setSingleStep(1)
        form_layout.addRow("模拟时长 (秒):", self.duration_spinbox)
        
        # 视图选择
        self.view_combo = QComboBox()
        self.view_combo.addItems(["3D视图", "侧视图", "俯视图"])
        self.view_combo.currentIndexChanged.connect(self.change_view)
        form_layout.addRow("视图选择:", self.view_combo)
        
        # 控制按钮
        buttons_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.on_start)
        buttons_layout.addWidget(self.start_button)
        
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.on_pause)
        self.pause_button.setEnabled(False)
        buttons_layout.addWidget(self.pause_button)
        
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.on_reset)
        buttons_layout.addWidget(self.reset_button)
        
        # 将表单和按钮组合到垂直布局中
        layout = QVBoxLayout()
        group_box = QGroupBox("参数控制")
        group_box.setLayout(form_layout)
        layout.addWidget(group_box)
        layout.addLayout(buttons_layout)
        
        # 添加伸展空间
        layout.addStretch(1)
        
        self.setLayout(layout)
        
    def on_parameter_changed(self):
        self.parameter_changed.emit()
        
    def on_start(self):
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.emit_start_signal()
        
    def on_pause(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.emit_pause_signal()
        
    def on_reset(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.emit_reset_signal()
        
    def change_view(self, index):
        # 这个方法将在主窗口中被重写
        pass
        
    # 以下方法将在主窗口中连接到相应的槽函数
    def emit_start_signal(self):
        pass
    
    def emit_pause_signal(self):
        pass
    
    def emit_reset_signal(self):
        pass
    
    def get_parameters(self):
        """获取当前参数设置"""
        return {
            'length': self.length_spinbox.value(),
            'mass': self.mass_spinbox.value(),
            'gravity': self.gravity_spinbox.value(),
            'damping': self.damping_spinbox.value(),
            'initial_angle': np.deg2rad(self.angle_spinbox.value()),
            'duration': self.duration_spinbox.value()
        }

class DataVisualizer(QWidget):
    """数据可视化组件"""
    def __init__(self, parent=None):
        super(DataVisualizer, self).__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 创建选项卡部件
        tabs = QTabWidget()
        
        # 创建角度-时间图表
        self.angle_time_canvas = MplCanvas(self, width=5, height=4)
        angle_time_widget = QWidget()
        angle_time_layout = QVBoxLayout()
        angle_time_layout.addWidget(self.angle_time_canvas)
        angle_time_widget.setLayout(angle_time_layout)
        tabs.addTab(angle_time_widget, "角度-时间")
        
        # 创建位置-时间图表
        self.pos_time_canvas = MplCanvas(self, width=5, height=4)
        pos_time_widget = QWidget()
        pos_time_layout = QVBoxLayout()
        pos_time_layout.addWidget(self.pos_time_canvas)
        pos_time_widget.setLayout(pos_time_layout)
        tabs.addTab(pos_time_widget, "位置-时间")
        
        # 创建相空间图表
        self.phase_space_canvas = MplCanvas(self, width=5, height=4)
        phase_space_widget = QWidget()
        phase_space_layout = QVBoxLayout()
        phase_space_layout.addWidget(self.phase_space_canvas)
        phase_space_widget.setLayout(phase_space_layout)
        tabs.addTab(phase_space_widget, "相空间")
        
        # 创建能量图表
        self.energy_canvas = MplCanvas(self, width=5, height=4)
        energy_widget = QWidget()
        energy_layout = QVBoxLayout()
        energy_layout.addWidget(self.energy_canvas)
        energy_widget.setLayout(energy_layout)
        tabs.addTab(energy_widget, "能量")
        
        layout.addWidget(tabs)
        self.setLayout(layout)
        
    def update_plots(self, results):
        """更新所有图表"""
        # 更新角度-时间图
        self.angle_time_canvas.axes.clear()
        self.angle_time_canvas.axes.plot(results['time'], results['angle'])
        self.angle_time_canvas.axes.set_xlabel('时间 (s)')
        self.angle_time_canvas.axes.set_ylabel('角度 (rad)')
        self.angle_time_canvas.axes.set_title('角度随时间变化')
        self.angle_time_canvas.axes.grid(True)
        self.angle_time_canvas.draw()
        
        # 更新位置-时间图
        self.pos_time_canvas.axes.clear()
        self.pos_time_canvas.axes.plot(results['time'], results['x_position'], 'r-', label='X位置')
        self.pos_time_canvas.axes.plot(results['time'], results['y_position'], 'b-', label='Y位置')
        self.pos_time_canvas.axes.set_xlabel('时间 (s)')
        self.pos_time_canvas.axes.set_ylabel('位置 (m)')
        self.pos_time_canvas.axes.set_title('位置随时间变化')
        self.pos_time_canvas.axes.legend()
        self.pos_time_canvas.axes.grid(True)
        self.pos_time_canvas.draw()
        
        # 更新相空间图
        self.phase_space_canvas.axes.clear()
        self.phase_space_canvas.axes.plot(results['angle'], results['angular_velocity'])
        self.phase_space_canvas.axes.set_xlabel('角度 (rad)')
        self.phase_space_canvas.axes.set_ylabel('角速度 (rad/s)')
        self.phase_space_canvas.axes.set_title('相空间轨迹')
        self.phase_space_canvas.axes.grid(True)
        self.phase_space_canvas.draw()
        
        # 更新能量图
        self.energy_canvas.axes.clear()
        self.energy_canvas.axes.plot(results['time'], results['kinetic_energy'], 'r-', label='动能')
        self.energy_canvas.axes.plot(results['time'], results['potential_energy'], 'g-', label='势能')
        self.energy_canvas.axes.plot(results['time'], results['total_energy'], 'b-', label='总能量')
        self.energy_canvas.axes.set_xlabel('时间 (s)')
        self.energy_canvas.axes.set_ylabel('能量 (J)')
        self.energy_canvas.axes.set_title('能量随时间变化')
        self.energy_canvas.axes.legend()
        self.energy_canvas.axes.grid(True)
        self.energy_canvas.draw()

class PendulumApp(QMainWindow):
    """单摆3D可视化主应用程序"""
    def __init__(self):
        super(PendulumApp, self).__init__()
        self.init_ui()
        self.init_simulation()
        
    def init_ui(self):
        # 设置窗口基本属性
        self.setWindowTitle('单摆精确测量3D可视化平台')
        self.resize(1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 左侧：控制面板
        self.control_panel = PendulumControlPanel()
        self.control_panel.parameter_changed.connect(self.on_parameter_changed)
        self.control_panel.emit_start_signal = self.start_simulation
        self.control_panel.emit_pause_signal = self.pause_simulation
        self.control_panel.emit_reset_signal = self.reset_simulation
        self.control_panel.change_view = self.change_view
        main_layout.addWidget(self.control_panel, 1)
        
        # 中间：3D可视化区域
        right_layout = QVBoxLayout()
        
        self.pendulum_3d = Pendulum3DWidget()
        right_layout.addWidget(self.pendulum_3d, 3)
        
        # 底部：数据可视化区域
        self.data_visualizer = DataVisualizer()
        right_layout.addWidget(self.data_visualizer, 2)
        
        main_layout.addLayout(right_layout, 4)
        
        central_widget.setLayout(main_layout)
        
        # 初始化定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        
    def init_simulation(self):
        """初始化模拟"""
        params = self.control_panel.get_parameters()
        
        self.pendulum = PendulumSimulation(
            length=params['length'],
            mass=params['mass'],
            gravity=params['gravity'],
            damping=params['damping'],
            initial_angle=params['initial_angle']
        )
        
        # 运行模拟计算
        self.results = self.pendulum.simulate(t_span=(0, params['duration']), t_points=500)
        
        # 显示图表
        self.data_visualizer.update_plots(self.results)
        
        # 初始化动画参数
        self.current_frame = 0
        self.total_frames = len(self.results['time'])
        self.frame_interval = 20  # 毫秒
        
        # 更新3D视图
        self.update_3d_view()
        
        # 状态更新
        periods, avg_period = self.pendulum.calculate_periods()
        g_calculated = self.pendulum.calculate_gravity()
        self.statusBar().showMessage(f'平均周期: {avg_period:.4f}s, 计算重力加速度: {g_calculated:.4f}m/s²')
        
    def on_parameter_changed(self):
        """参数改变时重置模拟"""
        self.reset_simulation()
        
    def start_simulation(self):
        """开始模拟动画"""
        if not self.timer.isActive():
            self.timer.start(self.frame_interval)
            self.statusBar().showMessage('模拟运行中...')
            
    def pause_simulation(self):
        """暂停模拟动画"""
        if self.timer.isActive():
            self.timer.stop()
            self.statusBar().showMessage('模拟已暂停')
            
    def reset_simulation(self):
        """重置模拟"""
        # 停止定时器
        if self.timer.isActive():
            self.timer.stop()
            
        # 清除轨迹
        self.pendulum_3d.clear_trajectory()
        
        # 重新初始化模拟
        self.init_simulation()
        
        self.statusBar().showMessage('模拟已重置')
        
    def update_simulation(self):
        """更新模拟状态"""
        # 更新当前帧
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0
            self.pendulum_3d.clear_trajectory()
            
        # 更新3D视图
        self.update_3d_view()
        
    def update_3d_view(self):
        """更新3D视图"""
        if self.current_frame < self.total_frames:
            # 获取当前位置
            x = self.results['x_position'][self.current_frame]
            y = 0  # 在2D模拟中，y坐标始终为0
            z = self.results['y_position'][self.current_frame]
            
            # 更新单摆位置
            self.pendulum_3d.update_pendulum(x, y, z, self.pendulum.length)
            
    def change_view(self, index):
        """更改视图角度"""
        if index == 0:  # 3D视图
            self.pendulum_3d.setCameraPosition(distance=10, azimuth=-65, elevation=20)
        elif index == 1:  # 侧视图
            self.pendulum_3d.setCameraPosition(distance=10, azimuth=-90, elevation=0)
        elif index == 2:  # 俯视图
            self.pendulum_3d.setCameraPosition(distance=10, azimuth=0, elevation=90)

if __name__ == '__main__':
    # 启用高DPI缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    window = PendulumApp()
    window.show()
    sys.exit(app.exec_()) 