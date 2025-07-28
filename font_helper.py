"""
中文字体支持模块
用于解决matplotlib、streamlit和plotly中的中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import os
import sys
import glob
from pathlib import Path
import numpy as np

def add_chinese_font_support():
    """
    为matplotlib添加中文字体支持
    """
    # 设置matplotlib参数
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    # 检查字体目录是否存在，如果不存在则创建
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    if not os.path.exists(fonts_dir):
        os.makedirs(fonts_dir)
    
    # 检查是否已经有中文字体
    font_files = glob.glob(os.path.join(fonts_dir, '*.ttf')) + \
                 glob.glob(os.path.join(fonts_dir, '*.ttc')) + \
                 glob.glob(os.path.join(fonts_dir, '*.otf'))
                 
    if not font_files:
        print("警告: 未在fonts目录中找到中文字体文件，可能会导致中文显示问题")
        print("请将中文字体文件(如SimHei.ttf)放入fonts目录")
    
    # 添加字体路径
    for font_file in font_files:
        try:
            fm.fontManager.addfont(font_file)
            print(f"已加载字体: {os.path.basename(font_file)}")
        except Exception as e:
            print(f"加载字体 {os.path.basename(font_file)} 失败: {e}")
    
    # 重新加载字体
    fm._rebuild()
    
    # 验证字体列表
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in available_fonts if any(name in f for name in 
                    ['SimHei', 'SimSun', 'Microsoft YaHei', 'WenQuanYi', 'Noto Sans CJK', 'Source Han Sans'])]
    
    if chinese_fonts:
        print(f"可用中文字体: {', '.join(chinese_fonts[:5])}" + ("..." if len(chinese_fonts) > 5 else ""))
        matplotlib.rcParams['font.sans-serif'] = chinese_fonts + matplotlib.rcParams['font.sans-serif']
    
    return chinese_fonts

def setup_plotly_chinese_fonts():
    """
    为Plotly设置中文字体
    """
    # Plotly不直接支持字体文件加载，而是使用网页字体
    # 返回适用于plotly的字体配置
    return {
        'family': 'SimHei, "Microsoft YaHei", "WenQuanYi Micro Hei", Arial, sans-serif',
        'size': 14,
    }

def test_font_support():
    """
    测试中文字体支持
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    ax.plot(x, y)
    ax.set_title('中文标题测试')
    ax.set_xlabel('横坐标 (角度)')
    ax.set_ylabel('纵坐标 (幅值)')
    
    print("生成测试图像...")
    
    # 保存测试图像
    test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'font_test.png')
    fig.savefig(test_image_path, dpi=100)
    plt.close(fig)
    
    print(f"测试图像已保存至: {test_image_path}")
    return test_image_path

if __name__ == "__main__":
    print("初始化中文字体支持...")
    fonts = add_chinese_font_support()
    print(f"找到 {len(fonts)} 个中文字体")
    test_font_support()
    print("测试完成") 