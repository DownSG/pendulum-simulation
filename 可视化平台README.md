# 单摆精确测量可视化平台

<p align="center">
  <img src="https://cdn.pixabay.com/photo/2019/02/02/13/15/physics-3971309_1280.jpg" width="400" alt="单摆物理实验">
</p>

## 项目概述

单摆精确测量可视化平台是一个用于物理教学和实验的高级工具，提供了单摆运动的精确模拟、数据分析和可视化功能。本平台基于精确的物理模型，考虑了空气阻力等非理想因素，可以用于学习单摆运动规律、测量重力加速度等物理实验。

## 主要特性

### 物理模拟
- 精确的非线性单摆方程求解
- 考虑空气阻力、阻尼等因素
- 高精度数值积分

### 数据分析
- 周期精确测量与计算
- 重力加速度测量实验
- 理论与实验数据对比分析
- 误差分析与计算

### 丰富的可视化界面
- 命令行模式
- Jupyter Notebook交互模式
- 基于Streamlit的Web应用
- 基于PyQt5的3D桌面应用

## 安装指南

本项目提供了多种使用方式，您可以根据需求选择安装不同的组件：

### 基本安装 (命令行和Jupyter Notebook)

```bash
pip install numpy scipy matplotlib pandas ipywidgets --proxy=http://127.0.0.1:7897
```

### Web应用安装

```bash
pip install streamlit numpy scipy matplotlib pandas pillow --proxy=http://127.0.0.1:7897
```

### 3D桌面应用安装

```bash
pip install PyQt5 pyqtgraph pyopengl numpy scipy matplotlib --proxy=http://127.0.0.1:7897
```

## 快速开始

### 命令行模式

```bash
python main.py
```
根据提示选择实验模式（1-4）。

### Jupyter Notebook模式

```bash
jupyter notebook pendulum_demo.ipynb
```

### Web应用模式

```bash
streamlit run pendulum_web_app.py
```

### 3D桌面应用模式

```bash
python pendulum_3d_visualization.py
```

## 实验模式

本平台支持以下实验模式：

### 1. 单摆基本模拟
- 模拟单摆运动并可视化
- 实时显示位置、角度、能量变化
- 计算周期和重力加速度

### 2. 理论与实验数据对比
- 比较理想理论模型与含有阻尼、噪声的"实验"数据
- 计算误差指标
- 可视化对比结果

### 3. 重力加速度测量实验
- 通过不同摆长测量周期
- 使用T²与L的线性关系估算重力加速度
- 计算测量不确定度

### 4. 交互式参数调整
- 实时调整物理参数并观察影响
- 支持在Jupyter或可视化界面中操作

## 项目结构

- `pendulum_simulation.py` - 单摆物理模拟核心类
- `data_analyzer.py` - 数据分析与可视化模块
- `main.py` - 命令行程序入口
- `pendulum_demo.ipynb` - Jupyter Notebook演示
- `pendulum_web_app.py` - Streamlit Web应用
- `pendulum_3d_visualization.py` - PyQt5 3D桌面应用
- `requirements.txt` - 项目依赖清单

## 屏幕截图

<p align="center">
  <em>命令行模式</em><br>
  <img src="https://cdn.pixabay.com/photo/2022/01/11/21/48/terminal-6931887_1280.png" width="400">
</p>

<p align="center">
  <em>Web应用模式</em><br>
  <img src="https://cdn.pixabay.com/photo/2022/05/08/03/10/dashboard-7181836_1280.png" width="400">
</p>

<p align="center">
  <em>3D桌面应用模式</em><br>
  <img src="https://cdn.pixabay.com/photo/2017/10/24/11/34/graph-2884204_1280.png" width="400">
</p>

## 教学应用

本平台适用于以下教学场景：

1. **物理基础教学**
   - 直观展示单摆运动规律
   - 可视化能量转换过程

2. **实验数据分析**
   - 学习实验数据处理方法
   - 理解误差来源与分析

3. **重力测量实验**
   - 学习重力加速度测量原理
   - 练习实验数据拟合分析

4. **计算物理学习**
   - 理解数值积分方法
   - 学习物理系统计算机模拟

## 开发与扩展

您可以通过以下方式扩展本平台：

1. 添加更多物理效应（如非均匀绳索、可变摆长等）
2. 添加双摆或多摆系统
3. 集成机器学习算法进行参数估计
4. 添加更多可视化工具和图表

## 详细文档

- [使用说明](./使用说明.md)
- [可视化平台安装指南](./可视化平台安装指南.md)

## 许可证

本项目采用MIT许可证 - 详见 LICENSE 文件

## 致谢

- 感谢numpy, scipy, matplotlib等开源库
- 感谢所有贡献者和使用者 