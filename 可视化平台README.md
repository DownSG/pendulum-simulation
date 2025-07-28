# Pendulum Visualization Platform

## Platform Overview

The Pendulum Visualization Platform is a visual analysis platform designed specifically for physics experiments. It provides support for pendulum-related experiments, especially focusing on the precise measurement of gravitational acceleration.

## Main Features

- **Intuitive Parameter Adjustment**: Adjust pendulum parameters through interactive controls to observe their effects on pendulum motion
- **Real-time Dynamic Animation**: Show pendulum motion in real time with dynamic trajectory tracking
- **Multi-dimensional Data Analysis**: Analyze angle, position, velocity, and energy changes over time
- **Precise Gravitational Acceleration Measurement**: Automatically calculate gravitational acceleration through period measurement

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python Version**: Python 3.8 or higher
- **Browser**: Chrome 80+, Firefox 72+, Edge 80+

## Running Instructions

1. Install dependencies:
   ```bash
   pip install streamlit numpy scipy matplotlib plotly pandas
   ```

2. Start the visualization platform:
   ```bash
   streamlit run pendulum_web_app_fixed.py
   ```

3. Open the browser and navigate to the displayed address (typically http://localhost:8501)

## Interface Introduction

- **Left Sidebar**: Parameter adjustment panel, you can modify the pendulum length, mass, gravity, damping, initial angle, etc.
- **Main Area**: According to the selected mode, it displays animation, data charts, or experimental results
- **Animation Area**: 
  - Interactive animation: You can play, pause, and drag the timeline
  - Frame sequence animation: Supports adjusting playback speed and frame number
- **Data Chart Area**: Displays various physical analysis charts, including:
  - Angle-time relationship
  - Phase space trajectory
  - Position-time curve
  - Energy change chart

## Example Usage

1. **Basic Pendulum Simulation**:
   - Adjust parameters on the left side, observe animation and data changes
   - View real-time physical data displayed on the right side

2. **Theory vs. Experimental Comparison**:
   - Set experimental parameters like noise level, damping coefficient
   - Compare theoretical and "experimental" data with noise
   - Analyze error indicators between theory and experiment

3. **Gravitational Acceleration Measurement**:
   - Set multiple pendulum lengths for automatic measurement
   - The system will automatically fit the T²-L relationship to calculate g
   - View measurement results and error analysis 