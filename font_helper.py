import os
import matplotlib
import matplotlib.font_manager as fm
import streamlit as st
import requests
from pathlib import Path
import tempfile

# 开源中文字体URL
NOTO_SANS_SC_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf"

def download_font_file(url, save_path):
    """下载字体文件到指定路径"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"下载字体文件失败: {e}")
        return False

def setup_chinese_font():
    """配置中文字体支持"""
    # 在临时目录中创建字体文件
    font_path = os.path.join(tempfile.gettempdir(), "NotoSansSC-Regular.otf")
    font_exists = os.path.exists(font_path)
    
    # 如果字体不存在，下载它
    if not font_exists:
        st.info("正在下载中文字体...")
        font_exists = download_font_file(NOTO_SANS_SC_URL, font_path)
    
    # 如果字体现在存在，注册它
    if font_exists:
        # 添加字体文件
        font_prop = fm.FontProperties(fname=font_path)
        
        # 配置matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        # 打印已加载的字体
        loaded_fonts = [f.name for f in fm.fontManager.ttflist]
        print(f"已加载字体: {loaded_fonts}")
        
        return font_prop
    else:
        # 尝试使用系统字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
        return None

def apply_chinese_font_to_figure(fig, font_prop=None):
    """将中文字体应用到图表"""
    if font_prop is None:
        # 尝试使用系统中可能的中文字体
        font_families = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
    else:
        font_families = [font_prop.get_name()]
    
    # 应用到所有文本对象
    for text_obj in fig.findobj(matplotlib.text.Text):
        if font_prop:
            text_obj.set_fontproperties(font_prop)
        else:
            text_obj.set_fontfamily(font_families)
    
    return fig

def apply_chinese_font_to_axes(ax, font_prop=None):
    """将中文字体应用到坐标轴"""
    if ax is None:
        return
    
    if font_prop:
        # 应用到文本
        for text in ax.texts:
            text.set_fontproperties(font_prop)
            
        # 应用到标题和标签
        ax.set_title(ax.get_title(), fontproperties=font_prop)
        ax.set_xlabel(ax.get_xlabel(), fontproperties=font_prop)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=font_prop)
    else:
        # 尝试使用系统中可能的中文字体
        font_families = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
        
        # 应用到文本
        for text in ax.texts:
            text.set_fontfamily(font_families)
            
        # 应用到标题和标签
        ax.set_title(ax.get_title(), fontfamily=font_families[0])
        ax.set_xlabel(ax.get_xlabel(), fontfamily=font_families[0])
        ax.set_ylabel(ax.get_ylabel(), fontfamily=font_families[0]) 